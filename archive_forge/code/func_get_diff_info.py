import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def get_diff_info(a_repo, start_rev, end_rev):
    """Get only the info for new revisions between the two revisions

    This lets us figure out what has actually changed between 2 revisions.
    """
    with ui.ui_factory.nested_progress_bar() as pb, a_repo.lock_read():
        graph = a_repo.get_graph()
        trace.note('getting ancestry diff')
        ancestry = graph.find_difference(start_rev, end_rev)[1]
        revs, canonical_committer = get_revisions_and_committers(a_repo, ancestry)
    return collapse_by_person(revs, canonical_committer)