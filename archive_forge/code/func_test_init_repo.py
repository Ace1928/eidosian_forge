import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_init_repo(self):
    output, error = self.run_bzr(['init', '--format=git', 'bla.git'])
    self.assertEqual(error, '')
    self.assertEqual(output, 'Created a standalone tree (format: git)\n')