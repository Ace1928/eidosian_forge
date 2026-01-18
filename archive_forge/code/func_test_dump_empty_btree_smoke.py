from breezy import tests
from breezy.bzr import btree_index
from breezy.tests import http_server
def test_dump_empty_btree_smoke(self):
    self.create_sample_empty_btree_index()
    out, err = self.run_bzr('dump-btree test.btree')
    self.assertEqualDiff('', out)