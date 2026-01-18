import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test__dump_index_simple(self):
    di = self._gc_module.DeltaIndex()
    di.add_source(_text1, 0)
    self.assertFalse(di._has_index())
    self.assertEqual(None, di._dump_index())
    _ = di.make_delta(_text1)
    self.assertTrue(di._has_index())
    hash_list, entry_list = di._dump_index()
    self.assertEqual(16, len(hash_list))
    self.assertEqual(68, len(entry_list))
    just_entries = [(idx, text_offset, hash_val) for idx, (text_offset, hash_val) in enumerate(entry_list) if text_offset != 0 or hash_val != 0]
    rabin_hash = self._gc_module._rabin_hash
    self.assertEqual([(8, 16, rabin_hash(_text1[1:17])), (25, 48, rabin_hash(_text1[33:49])), (34, 32, rabin_hash(_text1[17:33])), (47, 64, rabin_hash(_text1[49:65]))], just_entries)
    for entry_idx, text_offset, hash_val in just_entries:
        self.assertEqual(entry_idx, hash_list[hash_val & 15])