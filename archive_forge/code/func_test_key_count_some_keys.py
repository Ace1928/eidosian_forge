from ... import errors, tests, transport
from .. import index as _mod_index
def test_key_count_some_keys(self):
    index, adapter = self.make_index(nodes=[((b'notprefix', b'key1'), b'data', ((),)), ((b'prefix', b'key1'), b'data1', ((),)), ((b'prefix', b'key2'), b'data2', (((b'prefix', b'key1'),),))])
    self.assertEqual(2, adapter.key_count())