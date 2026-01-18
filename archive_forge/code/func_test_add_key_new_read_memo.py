import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_add_key_new_read_memo(self):
    """Adding a key with an uncached read_memo new to this batch adds that
        read_memo to the list of memos to fetch.
        """
    read_memo = ('fake index', 100, 50)
    locations = {('key',): (read_memo + (None, None), None, None, None)}
    batcher = groupcompress._BatchingBlockFetcher(StubGCVF(), locations)
    total_size = batcher.add_key(('key',))
    self.assertEqual(50, total_size)
    self.assertEqual([('key',)], batcher.keys)
    self.assertEqual([read_memo], batcher.memos_to_get)