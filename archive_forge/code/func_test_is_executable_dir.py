from breezy.tests import per_tree
from breezy.tests.features import SymlinkFeature
def test_is_executable_dir(self):
    tree = self.get_tree_with_subdirs_and_all_supported_content_types(False)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual(False, tree.is_executable('1top-dir'))