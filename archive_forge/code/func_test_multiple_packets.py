from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_multiple_packets(self):
    pktlines = []
    parser = PktLineParser(pktlines.append)
    parser.parse(b'0005z0006aba')
    self.assertEqual(pktlines, [b'z', b'ab'])
    self.assertEqual(b'a', parser.get_tail())