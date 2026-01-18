from breezy import errors, tests, workingtree
def test_knit1_failure(self):
    base_tree, sub_tree = self.make_trees(format='knit')
    self.assertRaises(errors.SubsumeTargetNeedsUpgrade, base_tree.subsume, sub_tree)