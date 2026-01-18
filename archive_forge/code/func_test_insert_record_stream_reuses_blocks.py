import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_insert_record_stream_reuses_blocks(self):
    vf = self.make_test_vf(True, dir='source')
    vf.insert_record_stream(self.grouped_stream([b'a', b'b', b'c', b'd']))
    vf.insert_record_stream(self.grouped_stream([b'e', b'f', b'g', b'h'], first_parents=((b'd',),)))
    block_bytes = {}
    stream = vf.get_record_stream([(r.encode(),) for r in 'abcdefgh'], 'unordered', False)
    num_records = 0
    for record in stream:
        if record.key in [(b'a',), (b'e',)]:
            self.assertEqual('groupcompress-block', record.storage_kind)
        else:
            self.assertEqual('groupcompress-block-ref', record.storage_kind)
        block_bytes[record.key] = record._manager._block._z_content
        num_records += 1
    self.assertEqual(8, num_records)
    for r in 'abcd':
        key = (r.encode(),)
        self.assertIs(block_bytes[key], block_bytes[b'a',])
        self.assertNotEqual(block_bytes[key], block_bytes[b'e',])
    for r in 'efgh':
        key = (r.encode(),)
        self.assertIs(block_bytes[key], block_bytes[b'e',])
        self.assertNotEqual(block_bytes[key], block_bytes[b'a',])
    vf2 = self.make_test_vf(True, dir='target')
    keys = [(r.encode(),) for r in 'abcdefgh']

    def small_size_stream():
        for record in vf.get_record_stream(keys, 'groupcompress', False):
            record._manager._full_enough_block_size = record._manager._block._content_length
            yield record
    vf2.insert_record_stream(small_size_stream())
    stream = vf2.get_record_stream(keys, 'groupcompress', False)
    vf2.writer.end()
    num_records = 0
    for record in stream:
        num_records += 1
        self.assertEqual(block_bytes[record.key], record._manager._block._z_content)
    self.assertEqual(8, num_records)