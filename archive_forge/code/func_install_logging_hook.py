from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def install_logging_hook(self, prefix):
    """Add a hook that logs calls made to it.

        :returns: the list that the calls will be appended to.
        """
    hook_calls = []
    _mod_branch.Branch.hooks.install_named_hook(prefix + '_change_branch_tip', hook_calls.append, None)
    return hook_calls