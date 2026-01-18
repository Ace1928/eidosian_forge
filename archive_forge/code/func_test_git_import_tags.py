import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_git_import_tags(self):
    r = GitRepo.init('a', mkdir=True)
    self.build_tree(['a/file'])
    r.stage('file')
    cid = r.do_commit(ref=b'refs/heads/abranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
    r[b'refs/tags/atag'] = cid
    self.run_bzr(['git-import', '--colocated', 'a', 'b'])
    self.assertEqual({'.bzr'}, set(os.listdir('b')))
    b = ControlDir.open('b')
    self.assertEqual(['abranch'], b.branch_names())
    self.assertEqual(['atag'], list(b.open_branch('abranch').tags.get_tag_dict().keys()))