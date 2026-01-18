import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_local_signing_key(self):
    r = GitRepo.init('gitr', mkdir=True)
    self.build_tree_contents([('gitr/.git/config', '[user]\n  email = some@example.com\n  name = Test User\n  signingkey = D729A457\n')])
    out, err = self.run_bzr(['config', '-d', 'gitr', 'gpg_signing_key'])
    self.assertEqual(out, 'D729A457\n')
    self.assertEqual(err, '')