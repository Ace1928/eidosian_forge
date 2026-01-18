from ... import errors, tests, transport
from .. import index as _mod_index
def test_key_count_two(self):
    index = self.make_index(nodes=[((b'name',), b''), ((b'foo',), b'')])
    self.assertEqual(2, index.key_count())