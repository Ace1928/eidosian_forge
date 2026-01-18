from ... import errors, tests, transport
from .. import index as _mod_index
def test_lookup_key_via_location_buffers(self):
    index = self.make_index()
    del index._transport._activity[:]
    result = index._lookup_keys_via_location([(index._size // 2, (b'missing',))])
    self.assertEqual([('readv', 'index', [(30, 30), (0, 200)], True, 60)], index._transport._activity)
    self.assertEqual([((index._size // 2, (b'missing',)), False)], result)
    self.assertIsNot(None, index._nodes)
    self.assertEqual([], index._parsed_byte_map)