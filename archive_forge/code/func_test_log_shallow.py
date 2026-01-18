import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_log_shallow(self):
    output, error = self.run_bzr(['log', 'gitr'], retcode=3)
    self.assertEqual(error, 'brz: ERROR: Further revision history missing.\n')
    self.assertEqual(output, '------------------------------------------------------------\nrevision-id: git-v1:' + self.repo.head().decode('ascii') + '\ngit commit: ' + self.repo.head().decode('ascii') + '\ncommitter: Somebody <user@example.com>\ntimestamp: Mon 2018-05-14 20:36:05 +0000\nmessage:\n  message\n')