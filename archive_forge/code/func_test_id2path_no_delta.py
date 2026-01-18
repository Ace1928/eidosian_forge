from .... import branch, errors, osutils, tests
from ....bzr import inventory
from .. import revision_store
from . import FastimportFeature
def test_id2path_no_delta(self):
    basis_inv = self.make_trivial_basis_inv()
    shim = revision_store._TreeShim(repo=None, basis_inv=basis_inv, inv_delta=[], content_provider=None)
    self.assertEqual('', shim.id2path(b'TREE_ROOT'))
    self.assertEqual('foo', shim.id2path(b'foo-id'))
    self.assertEqual('bar', shim.id2path(b'bar-id'))
    self.assertEqual('bar/baz', shim.id2path(b'baz-id'))
    self.assertRaises(errors.NoSuchId, shim.id2path, b'qux-id')