import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_manager_default_compressor_settings(self):
    locations, old_block = self.make_block(self._texts)
    manager = groupcompress._LazyGroupContentManager(old_block)
    gcvf = groupcompress.GroupCompressVersionedFiles
    self.assertIs(None, manager._compressor_settings)
    self.assertEqual(gcvf._DEFAULT_COMPRESSOR_SETTINGS, manager._get_compressor_settings())