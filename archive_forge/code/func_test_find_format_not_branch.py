from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_find_format_not_branch(self):
    dir = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
    self.assertRaises(errors.NotBranchError, _mod_bzrbranch.BranchFormatMetadir.find_format, dir)