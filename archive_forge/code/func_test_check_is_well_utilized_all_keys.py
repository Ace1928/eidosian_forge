import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_check_is_well_utilized_all_keys(self):
    block, manager = self.make_block_and_full_manager(self._texts)
    self.assertFalse(manager.check_is_well_utilized())
    manager._full_enough_block_size = block._content_length
    self.assertTrue(manager.check_is_well_utilized())
    manager._full_enough_block_size = block._content_length + 1
    self.assertFalse(manager.check_is_well_utilized())
    manager._full_enough_mixed_block_size = block._content_length
    self.assertFalse(manager.check_is_well_utilized())