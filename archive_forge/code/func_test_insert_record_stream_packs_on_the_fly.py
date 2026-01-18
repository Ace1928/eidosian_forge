import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_insert_record_stream_packs_on_the_fly(self):
    vf = self.make_test_vf(True, dir='source')
    vf.insert_record_stream(self.grouped_stream([b'a', b'b', b'c', b'd']))
    vf.insert_record_stream(self.grouped_stream([b'e', b'f', b'g', b'h'], first_parents=((b'd',),)))
    vf2 = self.make_test_vf(True, dir='target')
    keys = [(r.encode(),) for r in 'abcdefgh']
    vf2.insert_record_stream(vf.get_record_stream(keys, 'groupcompress', False))
    stream = vf2.get_record_stream(keys, 'groupcompress', False)
    vf2.writer.end()
    num_records = 0
    block = None
    for record in stream:
        num_records += 1
        if block is None:
            block = record._manager._block
        else:
            self.assertIs(block, record._manager._block)
    self.assertEqual(8, num_records)