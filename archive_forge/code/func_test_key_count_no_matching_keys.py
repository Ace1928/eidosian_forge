from ... import errors, tests, transport
from .. import index as _mod_index
def test_key_count_no_matching_keys(self):
    index, adapter = self.make_index(nodes=[((b'notprefix', b'key1'), b'data', ((),))])
    self.assertEqual(0, adapter.key_count())