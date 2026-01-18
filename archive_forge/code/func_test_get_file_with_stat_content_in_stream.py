from .... import branch, errors, osutils, tests
from ....bzr import inventory
from .. import revision_store
from . import FastimportFeature
def test_get_file_with_stat_content_in_stream(self):
    basis_inv = self.make_trivial_basis_inv()

    def content_provider(file_id):
        return b'content of\n' + file_id + b'\n'
    shim = revision_store._TreeShim(repo=None, basis_inv=basis_inv, inv_delta=[], content_provider=content_provider)
    f_obj, stat_val = shim.get_file_with_stat('bar/baz')
    self.assertIs(None, stat_val)
    self.assertEqualDiff(b'content of\nbaz-id\n', f_obj.read())