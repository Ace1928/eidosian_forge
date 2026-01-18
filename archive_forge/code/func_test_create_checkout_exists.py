from breezy.tests import per_branch
def test_create_checkout_exists(self):
    """We shouldn't fail if the directory already exists."""
    tree1 = self.make_branch_and_tree('base')
    self.build_tree(['checkout/'])
    tree2 = tree1.branch.create_checkout('checkout', lightweight=True)