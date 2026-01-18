from breezy.tests import per_branch
def test_create_lightweight_checkout(self):
    """We should be able to make a lightweight checkout."""
    tree1 = self.make_branch_and_tree('base')
    tree2 = tree1.branch.create_checkout('checkout', lightweight=True)
    self.assertNotEqual(tree1.basedir, tree2.basedir)
    self.assertEqual(tree1.branch.base, tree2.branch.base)