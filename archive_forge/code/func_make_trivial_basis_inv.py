from .... import branch, errors, osutils, tests
from ....bzr import inventory
from .. import revision_store
from . import FastimportFeature
def make_trivial_basis_inv(self):
    basis_inv = inventory.Inventory(b'TREE_ROOT')
    self.invAddEntry(basis_inv, 'foo', b'foo-id')
    self.invAddEntry(basis_inv, 'bar/', b'bar-id')
    self.invAddEntry(basis_inv, 'bar/baz', b'baz-id')
    return basis_inv