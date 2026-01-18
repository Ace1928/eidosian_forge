import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_diff_format(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    tree.add(['a'])
    output, error = self.run_bzr(['diff', '--format=git'], retcode=1)
    self.assertEqual(error, '')
    from dulwich import __version__ as dulwich_version
    if dulwich_version < (0, 19, 12):
        self.assertEqual(output, 'diff --git /dev/null b/a\nold mode 0\nnew mode 100644\nindex 0000000..c197bd8 100644\n--- /dev/null\n+++ b/a\n@@ -0,0 +1 @@\n+contents of a\n')
    else:
        self.assertEqual(output, 'diff --git a/a b/a\nold file mode 0\nnew file mode 100644\nindex 0000000..c197bd8 100644\n--- /dev/null\n+++ b/a\n@@ -0,0 +1 @@\n+contents of a\n')