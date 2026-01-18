from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
def test_allow_reserved_false(self):
    w = Weave('name', allow_reserved=False)
    w.add_lines(b'name:', [], TEXT_1)
    self.assertRaises(errors.ReservedId, w.get_lines, b'name:')