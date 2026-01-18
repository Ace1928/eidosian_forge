from six.moves import range
import sys
from pyu2f import apdu
from pyu2f import errors
def testResponseApduParse(self):
    resp = apdu.ResponseApdu(bytearray([5, 4, 144, 0]))
    self.assertEqual(resp.body, bytearray([5, 4]))
    self.assertEqual(resp.sw1, 144)
    self.assertEqual(resp.sw2, 0)
    self.assertTrue(resp.IsSuccess())