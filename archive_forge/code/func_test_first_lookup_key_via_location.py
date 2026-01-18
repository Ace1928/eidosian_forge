from ... import errors, tests, transport
from .. import index as _mod_index
def test_first_lookup_key_via_location(self):
    nodes = []
    index = self.make_index(nodes=self.make_nodes(64))
    del index._transport._activity[:]
    start_lookup = index._size // 2
    result = index._lookup_keys_via_location([(start_lookup, (b'40missing',))])
    self.assertEqual([('readv', 'index', [(start_lookup, 800), (0, 200)], True, index._size)], index._transport._activity)
    self.assertEqual([((start_lookup, (b'40missing',)), False)], result)
    self.assertIs(None, index._nodes)
    self.assertEqual([(0, 4008), (5046, 8996)], index._parsed_byte_map)
    self.assertEqual([((), self.make_key(26)), (self.make_key(31), self.make_key(48))], index._parsed_key_map)