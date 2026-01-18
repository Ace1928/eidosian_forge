import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_get_record_stream_max_bytes_to_index_default(self):
    vf = self.make_test_vf(True, dir='source')
    vf.add_lines((b'a',), (), [b'lines\n'])
    vf.writer.end()
    record = next(vf.get_record_stream([(b'a',)], 'unordered', True))
    self.assertEqual(vf._DEFAULT_COMPRESSOR_SETTINGS, record._manager._get_compressor_settings())