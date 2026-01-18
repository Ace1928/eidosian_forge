import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_gather_location_heavy_checkout(self):
    tree = self.make_branch_and_tree('tree')
    checkout = tree.branch.create_checkout('checkout')
    self.assertEqual([('checkout root', checkout.controldir.root_transport.base), ('checkout of branch', tree.controldir.root_transport.base)], self.gather_tree_location_info(checkout))
    light_checkout = checkout.branch.create_checkout('light_checkout', lightweight=True)
    self.assertEqual([('light checkout root', light_checkout.controldir.root_transport.base), ('checkout root', checkout.controldir.root_transport.base), ('checkout of branch', tree.controldir.root_transport.base)], self.gather_tree_location_info(light_checkout))