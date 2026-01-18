from breezy import tests
from breezy.bzr import btree_index
from breezy.tests import http_server
def test_dump_btree_raw_smoke(self):
    self.create_sample_btree_index()
    out, err = self.run_bzr('dump-btree test.btree --raw')
    self.assertEqualDiff('Root node:\nB+Tree Graph Index 2\nnode_ref_lists=1\nkey_elements=2\nlen=3\nrow_lengths=1\n\nPage 0\ntype=leaf\ntest\x00key1\x00ref\x00entry\x00value\ntest\x00key2\x00ref\x00entry2\x00value2\ntest2\x00key3\x00ref\x00entry3\x00value3\n\n', out)