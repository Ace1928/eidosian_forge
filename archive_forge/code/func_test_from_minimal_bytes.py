import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_from_minimal_bytes(self):
    block = groupcompress.GroupCompressBlock.from_bytes(b'gcb1z\n0\n0\n')
    self.assertIsInstance(block, groupcompress.GroupCompressBlock)
    self.assertIs(None, block._content)
    self.assertEqual(b'', block._z_content)
    block._ensure_content()
    self.assertEqual(b'', block._content)
    self.assertEqual(b'', block._z_content)
    block._ensure_content()