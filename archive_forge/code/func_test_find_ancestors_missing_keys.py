from ... import errors, tests, transport
from .. import index as _mod_index
def test_find_ancestors_missing_keys(self):
    key1 = (b'key-1',)
    key2 = (b'key-2',)
    key3 = (b'key-3',)
    key4 = (b'key-4',)
    index1 = self.make_index('12', ref_lists=1, nodes=[(key1, b'value', ([],)), (key2, b'value', ([key1],))])
    index2 = self.make_index('34', ref_lists=1, nodes=[(key3, b'value', ([key2],))])
    c_index = _mod_index.CombinedGraphIndex([index1, index2])
    parent_map, missing_keys = c_index.find_ancestry([key4], 0)
    self.assertEqual({}, parent_map)
    self.assertEqual({key4}, missing_keys)