from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_lightweight_format_checkout_to_tree(self, format=None):
    branch = self.make_branch('branch', format=format)
    checkout = branch.create_checkout('checkout', lightweight=True)
    tree = workingtree.WorkingTree.open('checkout')
    self.build_tree_contents([('checkout/file', b'foo\n')])
    tree.add(['file'])
    tree.commit('added file')
    self.run_bzr('reconfigure --tree', working_dir='checkout')
    tree = workingtree.WorkingTree.open('checkout')
    self.build_tree_contents([('checkout/file', b'bar\n')])
    self.check_file_contents('checkout/file', b'bar\n')
    self.run_bzr('revert', working_dir='checkout')
    self.check_file_contents('checkout/file', b'foo\n')