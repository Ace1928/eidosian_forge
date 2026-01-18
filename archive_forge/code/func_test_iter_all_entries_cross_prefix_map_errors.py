from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_all_entries_cross_prefix_map_errors(self):
    index, adapter = self.make_index(nodes=[((b'prefix', b'key1'), b'data1', (((b'prefixaltered', b'key2'),),))])
    self.assertRaises(_mod_index.BadIndexData, list, adapter.iter_all_entries())