from breezy import tests
from breezy.bzr import btree_index
from breezy.tests import http_server
def test_dump_empty_btree_raw_smoke(self):
    self.create_sample_empty_btree_index()
    out, err = self.run_bzr('dump-btree test.btree --raw')
    self.assertEqualDiff('Root node:\nB+Tree Graph Index 2\nnode_ref_lists=1\nkey_elements=2\nlen=0\nrow_lengths=\n\nPage 0\n(empty)\n', out)