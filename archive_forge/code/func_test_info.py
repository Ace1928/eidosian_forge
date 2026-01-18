import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_info(self):
    self.simple_commit()
    output, error = self.run_bzr(['info'])
    self.assertEqual(error, '')
    self.assertEqual(output, 'Standalone tree (format: git)\nLocation:\n            light checkout root: .\n  checkout of co-located branch: master\n')