import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_many_entries(self):
    lines = [b'type=leaf\n']
    for i in range(500):
        key_str = b'sha1:%04x%s' % (i, _hex_form[:36])
        key = (key_str,)
        lines.append(b'%s\x00\x00%d %d %d %d\n' % (key_str, i, i, i, i))
    data = b''.join(lines)
    leaf = self.module._parse_into_chk(data, 1, 0)
    self.assertEqual(24 - 7, leaf.common_shift)
    offsets = leaf._get_offsets()
    lst = [x // 2 for x in range(500)]
    expected_offsets = [x * 2 for x in range(128)] + [255] * 129
    self.assertEqual(expected_offsets, offsets)
    lst = lst[:255]
    self.assertEqual([bisect.bisect_left(lst, x) for x in range(0, 257)], offsets)
    for val in lst:
        self.assertEqual(lst.index(val), offsets[val])
    for idx, key in enumerate(leaf.all_keys()):
        self.assertEqual(b'%d' % idx, leaf[key][0].split()[0])