from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_255_does_not_include_newline(self):
    chars_used = set()
    for char_in in range(256):
        search_key = self.module._search_key_255(stuple(bytes([char_in])))
        chars_used.update([bytes([x]) for x in search_key])
    all_chars = {bytes([x]) for x in range(256)}
    unused_chars = all_chars.symmetric_difference(chars_used)
    self.assertEqual({b'\n'}, unused_chars)