from breezy.tests import TestNotApplicable
from breezy.tests.per_tree import TestCaseWithTree
from breezy.workingtree import SettingFileIdUnsupported
def make_tree_with_default_root_id(self):
    tree = self.make_branch_and_tree('tree')
    return self._convert_tree(tree)