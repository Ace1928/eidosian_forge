from .... import branch, errors, osutils, tests
from ....bzr import inventory
from .. import revision_store
from . import FastimportFeature
def test_get_symlink_target(self):
    basis_inv = self.make_trivial_basis_inv()
    ie = inventory.make_entry('symlink', 'link', b'TREE_ROOT', b'link-id')
    ie.symlink_target = 'link-target'
    basis_inv.add(ie)
    shim = revision_store._TreeShim(repo=None, basis_inv=basis_inv, inv_delta=[], content_provider=None)
    self.assertEqual('link-target', shim.get_symlink_target('link'))