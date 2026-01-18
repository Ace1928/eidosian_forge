from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_default_format(self):
    self.assertIsInstance(_mod_branch.format_registry.get_default(), _mod_bzrbranch.BzrBranchFormat7)