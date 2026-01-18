from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_all_entries_simple_2_elements(self):
    index = self.make_index(key_elements=2, nodes=[((b'name', b'surname'), b'data', ())])
    self.assertEqual([(index, (b'name', b'surname'), b'data')], list(index.iter_all_entries()))