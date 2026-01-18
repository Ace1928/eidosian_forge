from breezy.osutils import basename
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_dotbzr_is_control_in_subdir(self):
    tree = self.make_branch_and_tree('subdir')
    self.validate_tree_is_controlfilename(tree)