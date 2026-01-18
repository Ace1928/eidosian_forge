from breezy import tests
from breezy.bzr import btree_index
from breezy.tests import http_server
def test_dump_btree_smoke(self):
    self.create_sample_btree_index()
    out, err = self.run_bzr('dump-btree test.btree')
    self.assertEqualDiff("(('test', 'key1'), 'value', ((('ref', 'entry'),),))\n(('test', 'key2'), 'value2', ((('ref', 'entry2'),),))\n(('test2', 'key3'), 'value3', ((('ref', 'entry3'),),))\n", out)