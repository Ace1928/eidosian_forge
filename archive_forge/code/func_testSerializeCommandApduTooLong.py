from six.moves import range
import sys
from pyu2f import apdu
from pyu2f import errors
def testSerializeCommandApduTooLong(self):
    self.assertRaises(errors.InvalidCommandError, apdu.CommandApdu, 0, 1, 3, 4, bytearray((0 for x in range(0, 65536))))