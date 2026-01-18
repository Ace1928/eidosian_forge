from ... import errors, tests, transport
from .. import index as _mod_index
def test_key_count_one(self):
    index = self.make_index(nodes=[((b'name',), b'')])
    self.assertEqual(1, index.key_count())