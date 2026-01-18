from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_invalid_no_space(self):
    self.assertBytesToTextKeyRaises(b'file:file-id\nparent-id\nname\nrevision-id\nda39a3ee5e6b4b0d3255bfef95601890afd80709\n100\nN')