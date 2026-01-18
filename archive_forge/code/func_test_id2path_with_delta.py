from .... import branch, errors, osutils, tests
from ....bzr import inventory
from .. import revision_store
from . import FastimportFeature
def test_id2path_with_delta(self):
    basis_inv = self.make_trivial_basis_inv()
    foo_entry = inventory.make_entry('file', 'foo2', b'TREE_ROOT', b'foo-id')
    inv_delta = [('foo', 'foo2', b'foo-id', foo_entry), ('bar/baz', None, b'baz-id', None)]
    shim = revision_store._TreeShim(repo=None, basis_inv=basis_inv, inv_delta=inv_delta, content_provider=None)
    self.assertEqual('', shim.id2path(b'TREE_ROOT'))
    self.assertEqual('foo2', shim.id2path(b'foo-id'))
    self.assertEqual('bar', shim.id2path(b'bar-id'))
    self.assertRaises(errors.NoSuchId, shim.id2path, b'baz-id')