from six.moves import range
import sys
from pyu2f import apdu
from pyu2f import errors
def testResponseApduCheckSuccessInvalidKeyHandle(self):
    resp = apdu.ResponseApdu(bytearray([106, 128]))
    self.assertRaises(errors.InvalidKeyHandleError, resp.CheckSuccessOrRaise)