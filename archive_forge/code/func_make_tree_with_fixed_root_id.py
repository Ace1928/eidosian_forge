from breezy.tests import TestNotApplicable
from breezy.tests.per_tree import TestCaseWithTree
from breezy.workingtree import SettingFileIdUnsupported
def make_tree_with_fixed_root_id(self):
    tree = self.make_branch_and_tree('tree')
    if not tree.supports_setting_file_ids():
        self.assertRaises(SettingFileIdUnsupported, tree.set_root_id, b'custom-tree-root-id')
        self.skipTest('tree format does not support setting tree id')
    tree.set_root_id(b'custom-tree-root-id')
    return self._convert_tree(tree)