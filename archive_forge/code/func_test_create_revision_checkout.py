from breezy.tests import per_branch
def test_create_revision_checkout(self):
    """Test that we can create a checkout from an earlier revision."""
    tree1 = self.make_branch_and_tree('base')
    self.build_tree(['base/a'])
    tree1.add(['a'])
    rev1 = tree1.commit('first')
    self.build_tree(['base/b'])
    tree1.add(['b'])
    tree1.commit('second')
    tree2 = tree1.branch.create_checkout('checkout', revision_id=rev1)
    self.assertEqual(rev1, tree2.last_revision())
    self.assertPathExists('checkout/a')
    self.assertPathDoesNotExist('checkout/b')