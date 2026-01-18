from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_flush_empty(self):
    self._writer.flush()
    self.assertOutputEquals(b'')