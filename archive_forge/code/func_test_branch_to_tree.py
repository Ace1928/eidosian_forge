from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_branch_to_tree(self):
    branch = self.make_branch('branch')
    self.run_bzr('reconfigure --tree branch')
    tree = workingtree.WorkingTree.open('branch')