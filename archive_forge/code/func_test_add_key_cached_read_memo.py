import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_add_key_cached_read_memo(self):
    """Adding a key with a cached read_memo will not cause that read_memo
        to be added to the list to fetch.
        """
    read_memo = ('fake index', 100, 50)
    gcvf = StubGCVF()
    gcvf._group_cache[read_memo] = 'fake block'
    locations = {('key',): (read_memo + (None, None), None, None, None)}
    batcher = groupcompress._BatchingBlockFetcher(gcvf, locations)
    total_size = batcher.add_key(('key',))
    self.assertEqual(0, total_size)
    self.assertEqual([('key',)], batcher.keys)
    self.assertEqual([], batcher.memos_to_get)