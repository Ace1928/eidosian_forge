import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def find_credits(repository, revid):
    """Find the credits of the contributors to a revision.

    :return: tuple with (authors, documenters, artists, translators)
    """
    ret = {'documentation': {}, 'code': {}, 'art': {}, 'translation': {}, None: {}}
    with repository.lock_read():
        graph = repository.get_graph()
        ancestry = [r for r, ps in graph.iter_ancestry([revid]) if ps is not None and r != NULL_REVISION]
        revs = repository.get_revisions(ancestry)
        with ui.ui_factory.nested_progress_bar() as pb:
            iterator = zip(revs, repository.get_revision_deltas(revs))
            for i, (rev, delta) in enumerate(iterator):
                pb.update('analysing revisions', i, len(revs))
                if len(rev.parent_ids) > 1:
                    continue
                for c in set(classify_delta(delta)):
                    for author in rev.get_apparent_authors():
                        if author not in ret[c]:
                            ret[c][author] = 0
                        ret[c][author] += 1

    def sort_class(name):
        return [author for author, _ in sorted(ret[name].items(), key=classify_key)]
    return (sort_class('code'), sort_class('documentation'), sort_class('art'), sort_class('translation'))