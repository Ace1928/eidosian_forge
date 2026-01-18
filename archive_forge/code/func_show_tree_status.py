import sys
from . import delta as _mod_delta
from . import errors as errors
from . import hooks as _mod_hooks
from . import log, osutils
from . import revision as _mod_revision
from . import tsort
from .trace import mutter, warning
from .workingtree import ShelvingUnsupported
def show_tree_status(wt, specific_files=None, show_ids=False, to_file=None, show_pending=True, revision=None, short=False, verbose=False, versioned=False, classify=True, show_long_callback=_mod_delta.report_delta):
    """Display summary of changes.

    By default this compares the working tree to a previous revision.
    If the revision argument is given, summarizes changes between the
    working tree and another, or between two revisions.

    The result is written out as Unicode and to_file should be able
    to encode that.

    If showing the status of a working tree, extra information is included
    about unknown files, conflicts, and pending merges.

    :param specific_files: If set, a list of filenames whose status should be
        shown.  It is an error to give a filename that is not in the working
        tree, or in the working inventory or in the basis inventory.
    :param show_ids: If set, includes each file's id.
    :param to_file: If set, write to this file (default stdout.)
    :param show_pending: If set, write pending merges.
    :param revision: If None, compare latest revision with working tree
        If not None, it must be a RevisionSpec list.
        If one revision, compare with working tree.
        If two revisions, show status between first and second.
    :param short: If True, gives short SVN-style status lines.
    :param verbose: If True, show all merged revisions, not just
        the merge tips
    :param versioned: If True, only shows versioned files.
    :param classify: Add special symbols to indicate file kind.
    :param show_long_callback: A callback: message = show_long_callback(to_file, delta,
        show_ids, show_unchanged, indent, filter), only used with the long output
    """
    if to_file is None:
        to_file = sys.stdout
    with wt.lock_read():
        new_is_working_tree = True
        if revision is None:
            if wt.last_revision() != wt.branch.last_revision():
                warning("working tree is out of date, run 'brz update'")
            new = wt
            old = new.basis_tree()
        elif len(revision) > 0:
            try:
                old = revision[0].as_tree(wt.branch)
            except errors.NoSuchRevision as e:
                raise errors.CommandError(str(e))
            if len(revision) > 1 and revision[1].spec is not None:
                try:
                    new = revision[1].as_tree(wt.branch)
                    new_is_working_tree = False
                except errors.NoSuchRevision as e:
                    raise errors.CommandError(str(e))
            else:
                new = wt
        with old.lock_read(), new.lock_read():
            for hook in hooks['pre_status']:
                hook(StatusHookParams(old, new, to_file, versioned, show_ids, short, verbose, specific_files=specific_files))
            specific_files, nonexistents = _filter_nonexistent(specific_files, old, new)
            want_unversioned = not versioned
            reporter = _mod_delta._ChangeReporter(output_file=to_file, unversioned_filter=new.is_ignored, classify=classify)
            report_changes(to_file, old, new, specific_files, reporter, show_long_callback, short=short, want_unversioned=want_unversioned, show_ids=show_ids, classify=classify)
            if specific_files is not None:
                ignored_files = [specific for specific in specific_files if new.is_ignored(specific)]
                if len(ignored_files) > 0 and (not short):
                    to_file.write('ignored:\n')
                    prefix = ' '
                else:
                    prefix = 'I  '
                for ignored_file in ignored_files:
                    to_file.write('{} {}\n'.format(prefix, ignored_file))
            conflicts = new.conflicts()
            if specific_files is not None:
                conflicts = conflicts.select_conflicts(new, specific_files, ignore_misses=True, recurse=True)[1]
            if len(conflicts) > 0 and (not short):
                to_file.write('conflicts:\n')
            for conflict in conflicts:
                if short:
                    prefix = 'C  '
                else:
                    prefix = ' '
                to_file.write('{} {}\n'.format(prefix, conflict.describe()))
            if nonexistents and (not short):
                to_file.write('nonexistent:\n')
            for nonexistent in nonexistents:
                if short:
                    prefix = 'X  '
                else:
                    prefix = ' '
                to_file.write('{} {}\n'.format(prefix, nonexistent))
            if new_is_working_tree and show_pending:
                show_pending_merges(new, to_file, short, verbose=verbose)
            if nonexistents:
                raise errors.PathsDoNotExist(nonexistents)
            for hook in hooks['post_status']:
                hook(StatusHookParams(old, new, to_file, versioned, show_ids, short, verbose, specific_files=specific_files))