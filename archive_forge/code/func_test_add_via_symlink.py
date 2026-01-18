import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_via_symlink(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    self.make_branch_and_tree('source')
    self.build_tree(['source/top.txt'])
    os.symlink('source', 'link')
    out = self.run_bzr(['add', 'link/top.txt'])[0]
    self.assertEqual(out, 'adding top.txt\n')