from six.moves import range
import sys
from pyu2f import apdu
from pyu2f import errors
def testResponseApduParseNoBody(self):
    resp = apdu.ResponseApdu(bytearray([105, 133]))
    self.assertEqual(resp.sw1, 105)
    self.assertEqual(resp.sw2, 133)
    self.assertFalse(resp.IsSuccess())