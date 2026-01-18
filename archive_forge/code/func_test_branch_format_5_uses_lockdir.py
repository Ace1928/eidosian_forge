from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_branch_format_5_uses_lockdir(self):
    url = self.get_url()
    bdir = bzrdir.BzrDirMetaFormat1().initialize(url)
    bdir.create_repository()
    branch = BzrBranchFormat5().initialize(bdir)
    t = self.get_transport()
    self.log('branch instance is %r' % branch)
    self.assertTrue(isinstance(branch, BzrBranch5))
    self.assertIsDirectory('.', t)
    self.assertIsDirectory('.bzr/branch', t)
    self.assertIsDirectory('.bzr/branch/lock', t)
    branch.lock_write()
    self.addCleanup(branch.unlock)
    self.assertIsDirectory('.bzr/branch/lock/held', t)