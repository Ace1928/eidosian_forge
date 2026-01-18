import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_empty_dir(self):
    tree = self.make_branch_and_tree('.', format='git')
    self.build_tree(['a/', 'a/foo'])
    self.build_tree_contents([('.gitignore', 'foo\n')])
    tree.add(['.gitignore'])
    tree.commit('add ignore')
    output, error = self.run_bzr('st')
    self.assertEqual(output, '')
    self.assertEqual(error, '')