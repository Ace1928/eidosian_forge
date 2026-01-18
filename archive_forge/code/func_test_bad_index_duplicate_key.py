from ... import errors, tests, transport
from .. import index as _mod_index
def test_bad_index_duplicate_key(self):
    error = _mod_index.BadIndexDuplicateKey('foo', 'bar')
    self.assertEqual("The key 'foo' is already in index 'bar'.", str(error))