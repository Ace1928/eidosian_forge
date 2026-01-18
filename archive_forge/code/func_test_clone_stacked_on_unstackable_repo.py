from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_clone_stacked_on_unstackable_repo(self):
    repo = self.make_repository('a', format='dirstate-tags')
    control = repo.controldir
    branch = _mod_bzrbranch.BzrBranchFormat7().initialize(control)
    cloned_bzrdir = control.clone('cloned')