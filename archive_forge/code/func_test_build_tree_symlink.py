import os
from breezy import tests
from breezy.tests import features
def test_build_tree_symlink(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    self.build_tree_contents([('link@', 'target')])
    self.assertEqual('target', os.readlink('link'))