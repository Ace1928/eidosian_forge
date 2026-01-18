import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_add_source_max_bytes_to_index(self):
    di = self._gc_module.DeltaIndex()
    di._max_bytes_to_index = 3 * 16
    di.add_source(_text1, 0)
    di.add_source(_text3, 3)
    start2 = len(_text1) + 3
    hash_list, entry_list = di._dump_index()
    self.assertEqual(16, len(hash_list))
    self.assertEqual(67, len(entry_list))
    just_entries = sorted([(text_offset, hash_val) for text_offset, hash_val in entry_list if text_offset != 0 or hash_val != 0])
    rabin_hash = self._gc_module._rabin_hash
    self.assertEqual([(25, rabin_hash(_text1[10:26])), (50, rabin_hash(_text1[35:51])), (75, rabin_hash(_text1[60:76])), (start2 + 44, rabin_hash(_text3[29:45])), (start2 + 88, rabin_hash(_text3[73:89])), (start2 + 132, rabin_hash(_text3[117:133]))], just_entries)