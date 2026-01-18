import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_many_key_leaf(self):
    leaf = self.module._parse_into_chk(_multi_key_content, 1, 0)
    self.assertEqual(8, len(leaf))
    all_keys = leaf.all_keys()
    self.assertEqual(8, len(leaf.all_keys()))
    for idx, key in enumerate(all_keys):
        self.assertEqual(b'%d' % idx, leaf[key][0].split()[0])