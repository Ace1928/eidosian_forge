import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_git_refs_from_bzr(self):
    tree = self.make_branch_and_tree('a')
    self.build_tree(['a/file'])
    tree.add(['file'])
    revid = tree.commit(committer=b'Joe <joe@example.com>', message=b'Dummy')
    tree.branch.tags.set_tag('atag', revid)
    stdout, stderr = self.run_bzr(['git-refs', 'a'])
    self.assertEqual(stderr, '')
    self.assertTrue('refs/tags/atag -> ' in stdout)
    self.assertTrue('HEAD -> ' in stdout)