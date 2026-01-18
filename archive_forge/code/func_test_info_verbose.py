import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_info_verbose(self):
    self.simple_commit()
    output, error = self.run_bzr(['info', '-v'])
    self.assertEqual(error, '')
    self.assertTrue('Standalone tree (format: git)' in output)
    self.assertTrue('control: Local Git Repository' in output)
    self.assertTrue('branch: Local Git Branch' in output)
    self.assertTrue('repository: Git Repository' in output)