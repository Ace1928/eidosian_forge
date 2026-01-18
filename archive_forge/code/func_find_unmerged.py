from . import errors, log
def find_unmerged(local_branch, remote_branch, restrict='all', include_merged=None, backward=False, local_revid_range=None, remote_revid_range=None):
    """Find revisions from each side that have not been merged.

    :param local_branch: Compare the history of local_branch
    :param remote_branch: versus the history of remote_branch, and determine
        mainline revisions which have not been merged.
    :param restrict: ('all', 'local', 'remote') If 'all', we will return the
        unique revisions from both sides. If 'local', we will return None
        for the remote revisions, similarly if 'remote' we will return None for
        the local revisions.
    :param include_merged: Show mainline revisions only if False,
        all revisions otherwise.
    :param backward: Show oldest versions first when True, newest versions
        first when False.
    :param local_revid_range: Revision-id range for filtering local_branch
        revisions (lower bound, upper bound)
    :param remote_revid_range: Revision-id range for filtering remote_branch
        revisions (lower bound, upper bound)

    :return: A list of [(revno, revision_id)] for the mainline revisions on
        each side.
    """
    if include_merged is None:
        include_merged = False
    with local_branch.lock_read(), remote_branch.lock_read():
        return _find_unmerged(local_branch, remote_branch, restrict=restrict, include_merged=include_merged, backward=backward, local_revid_range=local_revid_range, remote_revid_range=remote_revid_range)