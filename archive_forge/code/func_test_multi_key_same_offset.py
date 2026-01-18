import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_multi_key_same_offset(self):
    leaf = self.module._parse_into_chk(_multi_key_same_offset, 1, 0)
    self.assertEqual(24, leaf.common_shift)
    offsets = leaf._get_offsets()
    lst = [8, 200, 205, 205, 205, 205, 206, 206]
    self.assertEqual([bisect.bisect_left(lst, x) for x in range(0, 257)], offsets)
    for val in lst:
        self.assertEqual(lst.index(val), offsets[val])
    for idx, key in enumerate(leaf.all_keys()):
        self.assertEqual(b'%d' % idx, leaf[key][0].split()[0])