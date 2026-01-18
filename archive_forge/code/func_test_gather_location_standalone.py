import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_gather_location_standalone(self):
    tree = self.make_branch_and_tree('tree')
    self.assertEqual([('branch root', tree.controldir.root_transport.base)], info.gather_location_info(tree.branch.repository, tree.branch, tree, control=tree.controldir))
    self.assertEqual([('branch root', tree.controldir.root_transport.base)], info.gather_location_info(tree.branch.repository, tree.branch, control=tree.controldir))
    return tree