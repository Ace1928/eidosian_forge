from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_nothing_empty(self):
    index = self.make_index()
    self.assertEqual([], list(index.iter_entries([])))