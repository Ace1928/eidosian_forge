from six.moves import range
import sys
from pyu2f import apdu
from pyu2f import errors
def testResponseApduParseInvalid(self):
    self.assertRaises(errors.InvalidResponseError, apdu.ResponseApdu, bytearray([5]))