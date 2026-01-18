from breezy import branch, errors, uncommit
from breezy.tests import per_branch
def capture_post_uncommit_hook(self, local, master, old_revno, old_revid, new_revno, new_revid):
    """Capture post uncommit hook calls to self.hook_calls.

        The call is logged, as is some state of the two branches.
        """
    if local:
        local_locked = local.is_locked()
        local_base = local.base
    else:
        local_locked = None
        local_base = None
    self.hook_calls.append(('post_uncommit', local_base, master.base, old_revno, old_revid, new_revno, new_revid, local_locked, master.is_locked()))