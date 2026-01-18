from breezy.tests import per_branch
def test_checkout_format_heavyweight(self):
    """Make sure the new heavy checkout uses the desired branch format."""
    a_branch = self.make_branch('branch')
    tree = a_branch.create_checkout('checkout', lightweight=False)
    expected_format = a_branch._get_checkout_format(lightweight=False)
    self.assertEqual(expected_format.get_branch_format().network_name(), tree.branch._format.network_name())