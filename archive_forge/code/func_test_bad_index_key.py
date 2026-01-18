from ... import errors, tests, transport
from .. import index as _mod_index
def test_bad_index_key(self):
    error = _mod_index.BadIndexKey('foo')
    self.assertEqual("The key 'foo' is not a valid key.", str(error))