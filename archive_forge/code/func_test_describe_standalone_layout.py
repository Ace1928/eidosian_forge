import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_describe_standalone_layout(self):
    tree = self.make_branch_and_tree('tree')
    self.assertEqual('Empty control directory', info.describe_layout())
    self.assertEqual('Unshared repository with trees and colocated branches', info.describe_layout(tree.branch.repository, control=tree.controldir))
    tree.branch.repository.set_make_working_trees(False)
    self.assertEqual('Unshared repository with colocated branches', info.describe_layout(tree.branch.repository, control=tree.controldir))
    self.assertEqual('Standalone branch', info.describe_layout(tree.branch.repository, tree.branch, control=tree.controldir))
    self.assertEqual('Standalone branchless tree', info.describe_layout(tree.branch.repository, None, tree, control=tree.controldir))
    self.assertEqual('Standalone tree', info.describe_layout(tree.branch.repository, tree.branch, tree, control=tree.controldir))
    tree.branch.bind(tree.branch)
    self.assertEqual('Bound branch', info.describe_layout(tree.branch.repository, tree.branch, control=tree.controldir))
    self.assertEqual('Checkout', info.describe_layout(tree.branch.repository, tree.branch, tree, control=tree.controldir))
    checkout = tree.branch.create_checkout('checkout', lightweight=True)
    self.assertEqual('Lightweight checkout', info.describe_layout(checkout.branch.repository, checkout.branch, checkout, control=tree.controldir))