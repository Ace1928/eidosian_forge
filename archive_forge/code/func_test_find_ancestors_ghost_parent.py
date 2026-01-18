from ... import errors, tests, transport
from .. import index as _mod_index
def test_find_ancestors_ghost_parent(self):
    key1 = (b'key-1',)
    key2 = (b'key-2',)
    key3 = (b'key-3',)
    key4 = (b'key-4',)
    index1 = self.make_index('12', ref_lists=1, nodes=[(key1, b'value', ([],)), (key2, b'value', ([key1],))])
    index2 = self.make_index('34', ref_lists=1, nodes=[(key4, b'value', ([key2, key3],))])
    c_index = _mod_index.CombinedGraphIndex([index1, index2])
    parent_map, missing_keys = c_index.find_ancestry([key4], 0)
    self.assertEqual({key4: (key2, key3), key2: (key1,), key1: ()}, parent_map)
    self.assertEqual({key3}, missing_keys)