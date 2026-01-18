from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def assertBytesToTextKeyRaises(self, bytes):
    self.assertRaises(Exception, self.module._bytes_to_text_key, bytes)