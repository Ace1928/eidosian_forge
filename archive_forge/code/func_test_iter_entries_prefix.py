from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_entries_prefix(self):
    index, adapter = self.make_index(key_elements=3, nodes=[((b'notprefix', b'foo', b'key1'), b'data', ((),)), ((b'prefix', b'prefix2', b'key1'), b'data1', ((),)), ((b'prefix', b'prefix2', b'key2'), b'data2', (((b'prefix', b'prefix2', b'key1'),),))])
    self.assertEqual({(index, (b'prefix2', b'key1'), b'data1', ((),)), (index, (b'prefix2', b'key2'), b'data2', (((b'prefix2', b'key1'),),))}, set(adapter.iter_entries_prefix([(b'prefix2', None)])))