import json
import sys
from pyu2f import errors
from pyu2f import model
def testClientDataInvalid(self):
    self.assertRaises(errors.InvalidModelError, model.ClientData, 'foobar', b'ABCD', 'somemachine')