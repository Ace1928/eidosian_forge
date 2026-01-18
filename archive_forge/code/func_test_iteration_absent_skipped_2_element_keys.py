from ... import errors, tests, transport
from .. import index as _mod_index
def test_iteration_absent_skipped_2_element_keys(self):
    index = self.make_index(1, key_elements=2, nodes=[((b'name', b'fin'), b'data', ([(b'ref', b'erence')],))])
    self.assertEqual([(index, (b'name', b'fin'), b'data', (((b'ref', b'erence'),),))], list(index.iter_all_entries()))
    self.assertEqual([(index, (b'name', b'fin'), b'data', (((b'ref', b'erence'),),))], list(index.iter_entries([(b'name', b'fin')])))
    self.assertEqual([], list(index.iter_entries([(b'ref', b'erence')])))