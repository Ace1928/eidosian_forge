import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test__check_rebuild_middle(self):
    locations, block = self.make_block(self._texts)
    manager = groupcompress._LazyGroupContentManager(block)
    self.add_key_to_manager((b'key4',), locations, block, manager)
    manager._check_rebuild_block()
    self.assertIsNot(block, manager._block)
    self.assertTrue(block._content_length > manager._block._content_length)
    for record in manager.get_record_stream():
        self.assertEqual((b'key4',), record.key)
        self.assertEqual(self._texts[record.key], record.get_bytes_as('fulltext'))