import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_multiple_files_in_unicode_cwd(self):
    """Adding multiple files in a non-ascii cwd, see lp:686611"""
    self.requireFeature(features.UnicodeFilenameFeature)
    self.make_branch_and_tree('§')
    self.build_tree(['§/a', '§/b'])
    out, err = self.run_bzr(['add', 'a', 'b'], working_dir='§')
    self.assertEqual(out, 'adding a\nadding b\n')
    self.assertEqual(err, '')