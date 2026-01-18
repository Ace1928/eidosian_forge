from ... import errors, tests, transport
from .. import index as _mod_index
def test_lookup_present_key_answers_without_io_when_map_permits(self):
    index = self.make_index(nodes=self.make_nodes(64))
    result = index._lookup_keys_via_location([(index._size // 2, (b'40',))])
    self.assertEqual([(0, 4008), (5046, 8996)], index._parsed_byte_map)
    self.assertEqual([((), self.make_key(26)), (self.make_key(31), self.make_key(48))], index._parsed_key_map)
    del index._transport._activity[:]
    result = index._lookup_keys_via_location([(4000, self.make_key(40))])
    self.assertEqual([((4000, self.make_key(40)), (index, self.make_key(40), self.make_value(40)))], result)
    self.assertEqual([], index._transport._activity)