import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_git_import_incremental(self):
    r = GitRepo.init('a', mkdir=True)
    self.build_tree(['a/file'])
    r.stage('file')
    r.do_commit(ref=b'refs/heads/abranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
    self.run_bzr(['git-import', '--colocated', 'a', 'b'])
    self.run_bzr(['git-import', '--colocated', 'a', 'b'])
    self.assertEqual({'.bzr'}, set(os.listdir('b')))
    b = ControlDir.open('b')
    self.assertEqual(['abranch'], b.branch_names())