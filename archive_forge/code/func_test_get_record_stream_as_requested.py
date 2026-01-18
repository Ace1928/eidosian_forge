import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_get_record_stream_as_requested(self):
    vf = self.make_test_vf(False, dir='source')
    vf.add_lines((b'a',), (), [b'lines\n'])
    vf.add_lines((b'b',), (), [b'lines\n'])
    vf.add_lines((b'c',), (), [b'lines\n'])
    vf.add_lines((b'd',), (), [b'lines\n'])
    vf.writer.end()
    keys = [record.key for record in vf.get_record_stream([(b'a',), (b'b',), (b'c',), (b'd',)], 'as-requested', False)]
    self.assertEqual([(b'a',), (b'b',), (b'c',), (b'd',)], keys)
    keys = [record.key for record in vf.get_record_stream([(b'b',), (b'a',), (b'd',), (b'c',)], 'as-requested', False)]
    self.assertEqual([(b'b',), (b'a',), (b'd',), (b'c',)], keys)
    vf2 = self.make_test_vf(False, dir='target')
    vf2.insert_record_stream(vf.get_record_stream([(b'b',), (b'a',), (b'd',), (b'c',)], 'as-requested', False))
    vf2.writer.end()
    keys = [record.key for record in vf2.get_record_stream([(b'a',), (b'b',), (b'c',), (b'd',)], 'as-requested', False)]
    self.assertEqual([(b'a',), (b'b',), (b'c',), (b'd',)], keys)
    keys = [record.key for record in vf2.get_record_stream([(b'b',), (b'a',), (b'd',), (b'c',)], 'as-requested', False)]
    self.assertEqual([(b'b',), (b'a',), (b'd',), (b'c',)], keys)