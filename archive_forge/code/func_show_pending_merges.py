import sys
from . import delta as _mod_delta
from . import errors as errors
from . import hooks as _mod_hooks
from . import log, osutils
from . import revision as _mod_revision
from . import tsort
from .trace import mutter, warning
from .workingtree import ShelvingUnsupported
def show_pending_merges(new, to_file, short=False, verbose=False):
    """Write out a display of pending merges in a working tree."""
    parents = new.get_parent_ids()
    if len(parents) < 2:
        return
    term_width = osutils.terminal_width()
    if term_width is not None:
        term_width = term_width - 1
    if short:
        first_prefix = 'P   '
        sub_prefix = 'P.   '
    else:
        first_prefix = '  '
        sub_prefix = '    '

    def show_log_message(rev, prefix):
        if term_width is None:
            width = term_width
        else:
            width = term_width - len(prefix)
        log_message = log_formatter.log_string(None, rev, width, prefix=prefix)
        to_file.write(log_message + '\n')
    pending = parents[1:]
    branch = new.branch
    last_revision = parents[0]
    if not short:
        if verbose:
            to_file.write('pending merges:\n')
        else:
            to_file.write('pending merge tips: (use -v to see all merge revisions)\n')
    graph = branch.repository.get_graph()
    other_revisions = [last_revision]
    log_formatter = log.LineLogFormatter(to_file)
    for merge in pending:
        try:
            rev = branch.repository.get_revision(merge)
        except errors.NoSuchRevision:
            to_file.write(first_prefix + '(ghost) ' + merge.decode('utf-8') + '\n')
            other_revisions.append(merge)
            continue
        show_log_message(rev, first_prefix)
        if not verbose:
            continue
        merge_extra = graph.find_unique_ancestors(merge, other_revisions)
        other_revisions.append(merge)
        merge_extra.discard(_mod_revision.NULL_REVISION)
        revisions = dict(branch.repository.iter_revisions(merge_extra))
        rev_id_iterator = _get_sorted_revisions(merge, merge_extra, branch.repository.get_parent_map(merge_extra))
        num, first, depth, eom = next(rev_id_iterator)
        if first != merge:
            raise AssertionError('Somehow we misunderstood how iter_topo_order works %s != %s' % (first, merge))
        for num, sub_merge, depth, eom in rev_id_iterator:
            rev = revisions[sub_merge]
            if rev is None:
                to_file.write(sub_prefix + '(ghost) ' + sub_merge.decode('utf-8') + '\n')
                continue
            show_log_message(revisions[sub_merge], sub_prefix)