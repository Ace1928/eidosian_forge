from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_installed_hooks_are_BranchHooks(self):
    """The installed hooks object should be a BranchHooks."""
    self.assertIsInstance(self._preserved_hooks[_mod_branch.Branch][1], _mod_branch.BranchHooks)