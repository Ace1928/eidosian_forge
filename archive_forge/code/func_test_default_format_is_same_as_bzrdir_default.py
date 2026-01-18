from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_default_format_is_same_as_bzrdir_default(self):
    self.assertEqual(_mod_branch.format_registry.get_default(), bzrdir.BzrDirFormat.get_default_format().get_branch_format())