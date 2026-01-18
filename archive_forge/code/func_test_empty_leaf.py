import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_empty_leaf(self):
    leaf = self.module._parse_into_chk(b'type=leaf\n', 1, 0)
    self.assertEqual(0, len(leaf))
    self.assertEqual([], leaf.all_items())
    self.assertEqual([], leaf.all_keys())
    self.assertFalse(('key',) in leaf)