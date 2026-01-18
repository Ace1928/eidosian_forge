import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_gather_location_bound(self):
    branch = self.make_branch('branch')
    bound_branch = self.make_branch('bound_branch')
    bound_branch.bind(branch)
    self.assertEqual([('branch root', bound_branch.controldir.root_transport.base), ('bound to branch', branch.controldir.root_transport.base)], info.gather_location_info(bound_branch.repository, bound_branch, control=bound_branch.controldir))