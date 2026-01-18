import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_add_key_duplicate_read_memo(self):
    """read_memos that occur multiple times in a batch will only be fetched
        once.
        """
    read_memo = ('fake index', 100, 50)
    locations = {('key1',): (read_memo + (0, 1), None, None, None), ('key2',): (read_memo + (1, 2), None, None, None)}
    batcher = groupcompress._BatchingBlockFetcher(StubGCVF(), locations)
    total_size = batcher.add_key(('key1',))
    total_size = batcher.add_key(('key2',))
    self.assertEqual(50, total_size)
    self.assertEqual([('key1',), ('key2',)], batcher.keys)
    self.assertEqual([read_memo], batcher.memos_to_get)