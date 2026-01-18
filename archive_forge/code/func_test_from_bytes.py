import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_from_bytes(self):
    locations, block = self.make_block(self._texts)
    manager = groupcompress._LazyGroupContentManager(block)
    self.add_key_to_manager((b'key1',), locations, block, manager)
    self.add_key_to_manager((b'key4',), locations, block, manager)
    wire_bytes = manager._wire_bytes()
    self.assertStartsWith(wire_bytes, b'groupcompress-block\n')
    manager = groupcompress._LazyGroupContentManager.from_bytes(wire_bytes)
    self.assertIsInstance(manager, groupcompress._LazyGroupContentManager)
    self.assertEqual(2, len(manager._factories))
    self.assertEqual(block._z_content, manager._block._z_content)
    result_order = []
    for record in manager.get_record_stream():
        result_order.append(record.key)
        text = self._texts[record.key]
        self.assertEqual(text, record.get_bytes_as('fulltext'))
    self.assertEqual([(b'key1',), (b'key4',)], result_order)