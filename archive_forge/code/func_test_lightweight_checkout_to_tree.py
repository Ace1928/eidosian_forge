from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_lightweight_checkout_to_tree(self):
    branch = self.make_branch('branch')
    checkout = branch.create_checkout('checkout', lightweight=True)
    self.run_bzr('reconfigure --tree checkout')