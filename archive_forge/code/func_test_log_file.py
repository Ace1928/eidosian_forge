import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_log_file(self):
    repo = GitRepo.init(self.test_dir)
    builder = tests.GitBranchBuilder()
    builder.set_file('a', b'text for a\n', False)
    r1 = builder.commit(b'Joe Foo <joe@foo.com>', 'First')
    builder.set_file('a', b'text 3a for a\n', False)
    r2a = builder.commit(b'Joe Foo <joe@foo.com>', 'Second a', base=r1)
    builder.set_file('a', b'text 3b for a\n', False)
    r2b = builder.commit(b'Joe Foo <joe@foo.com>', 'Second b', base=r1)
    builder.set_file('a', b'text 4 for a\n', False)
    builder.commit(b'Joe Foo <joe@foo.com>', 'Third', merge=[r2a], base=r2b)
    builder.finish()
    output, error = self.run_bzr(['log', '-n2', 'a'])
    self.assertEqual(error, '')
    self.assertIn('Second a', output)
    self.assertIn('Second b', output)
    self.assertIn('First', output)
    self.assertIn('Third', output)