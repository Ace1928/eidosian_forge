import os
from dulwich.repo import Repo as GitRepo
from ... import controldir, errors, urlutils
from ...tests import TestSkipped
from ...transport import get_transport
from .. import dir, tests, workingtree
def test_git_file(self):
    gitrepo = GitRepo.init('blah', mkdir=True)
    self.build_tree_contents([('foo/',), ('foo/.git', b'gitdir: ../blah/.git\n')])
    gd = controldir.ControlDir.open('foo')
    self.assertEqual(gd.control_url.rstrip('/'), urlutils.local_path_to_url(os.path.abspath(gitrepo.controldir())))