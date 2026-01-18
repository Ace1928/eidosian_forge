import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_max_bytes_to_index_default(self):
    vf = self.make_test_vf()
    gc = vf._make_group_compressor()
    self.assertEqual(vf._DEFAULT_MAX_BYTES_TO_INDEX, vf._max_bytes_to_index)
    if isinstance(gc, groupcompress.PyrexGroupCompressor):
        self.assertEqual(vf._DEFAULT_MAX_BYTES_TO_INDEX, gc._delta_index._max_bytes_to_index)