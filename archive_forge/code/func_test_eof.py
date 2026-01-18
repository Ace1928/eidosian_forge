from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_eof(self):
    self.rin.allow_read_past_eof = True
    BaseProtocolTests.test_eof(self)