import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_gather_location_controldir_only(self):
    bzrdir = self.make_controldir('.')
    self.assertEqual([('control directory', bzrdir.user_url)], info.gather_location_info(control=bzrdir))