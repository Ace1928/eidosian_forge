import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_git_refs_from_git(self):
    r = GitRepo.init('a', mkdir=True)
    self.build_tree(['a/file'])
    r.stage('file')
    cid = r.do_commit(ref=b'refs/heads/abranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
    r[b'refs/tags/atag'] = cid
    stdout, stderr = self.run_bzr(['git-refs', 'a'])
    self.assertEqual(stderr, '')
    self.assertEqual(stdout, 'refs/heads/abranch -> ' + cid.decode('ascii') + '\nrefs/tags/atag -> ' + cid.decode('ascii') + '\n')