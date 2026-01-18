import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test__rebuild_handles_compressor_settings(self):
    if not isinstance(groupcompress.GroupCompressor, groupcompress.PyrexGroupCompressor):
        raise tests.TestNotApplicable('pure-python compressor does not handle compressor_settings')
    locations, old_block = self.make_block(self._texts)
    manager = groupcompress._LazyGroupContentManager(old_block, get_compressor_settings=lambda: dict(max_bytes_to_index=32))
    gc = manager._make_group_compressor()
    self.assertEqual(32, gc._delta_index._max_bytes_to_index)
    self.add_key_to_manager((b'key3',), locations, old_block, manager)
    self.add_key_to_manager((b'key4',), locations, old_block, manager)
    action, last_byte, total_bytes = manager._check_rebuild_action()
    self.assertEqual('rebuild', action)
    manager._rebuild_block()
    new_block = manager._block
    self.assertIsNot(old_block, new_block)
    self.assertTrue(old_block._content_length < new_block._content_length)