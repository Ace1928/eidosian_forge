from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_caps_want_line(self):
    self.assertEqual((b'want bla', [b'la']), extract_want_line_capabilities(b'want bla la'))
    self.assertEqual((b'want bla', [b'la']), extract_want_line_capabilities(b'want bla la\n'))
    self.assertEqual((b'want bla', [b'la', b'la']), extract_want_line_capabilities(b'want bla la la'))