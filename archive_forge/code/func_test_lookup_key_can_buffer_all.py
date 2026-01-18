from ... import errors, tests, transport
from .. import index as _mod_index
def test_lookup_key_can_buffer_all(self):
    nodes = []
    for counter in range(64):
        nodes.append((self.make_key(counter), self.make_value(counter), ((self.make_key(counter + 20),),)))
    index = self.make_index(ref_lists=1, nodes=nodes)
    index_size = index._size
    index_center = index_size // 2
    result = index._lookup_keys_via_location([(index_center, (b'40',))])
    self.assertEqual([(0, 3890), (6444, 10274)], index._parsed_byte_map)
    self.assertEqual([((), self.make_key(25)), (self.make_key(37), self.make_key(52))], index._parsed_key_map)
    self.assertEqual([('readv', 'index', [(index_center, 800), (0, 200)], True, index_size)], index._transport._activity)
    del index._transport._activity[:]
    result = index._lookup_keys_via_location([(7000, self.make_key(40))])
    self.assertEqual([((7000, self.make_key(40)), (index, self.make_key(40), self.make_value(40), ((self.make_key(60),),)))], result)
    self.assertEqual([('get', 'index')], index._transport._activity)