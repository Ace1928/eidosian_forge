from breezy.tests import per_tree
from breezy.tests.features import SymlinkFeature
def test_is_executable_symlink(self):
    self.requireFeature(SymlinkFeature(self.test_dir))
    tree = self.get_tree_with_subdirs_and_all_content_types()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual(False, tree.is_executable('symlink'))