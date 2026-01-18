import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test__insert_record_stream_no_reuse_block(self):
    vf = self.make_test_vf(True, dir='source')
    vf.insert_record_stream(self.grouped_stream([b'a', b'b', b'c', b'd']))
    vf.insert_record_stream(self.grouped_stream([b'e', b'f', b'g', b'h'], first_parents=((b'd',),)))
    vf.writer.end()
    keys = [(r.encode(),) for r in 'abcdefgh']
    self.assertEqual(8, len(list(vf.get_record_stream(keys, 'unordered', False))))
    vf2 = self.make_test_vf(True, dir='target')
    list(vf2._insert_record_stream(vf.get_record_stream(keys, 'groupcompress', False), reuse_blocks=False))
    vf2.writer.end()
    stream = vf2.get_record_stream(keys, 'groupcompress', False)
    block = None
    for record in stream:
        if block is None:
            block = record._manager._block
        else:
            self.assertIs(block, record._manager._block)