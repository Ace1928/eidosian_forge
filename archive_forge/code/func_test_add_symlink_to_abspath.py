import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_symlink_to_abspath(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    self.make_branch_and_tree('tree')
    os.symlink(osutils.abspath('target'), 'tree/link')
    out = self.run_bzr(['add', 'tree/link'])[0]
    self.assertEqual(out, 'adding link\n')