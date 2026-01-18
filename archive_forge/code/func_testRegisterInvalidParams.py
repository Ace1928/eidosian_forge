import sys
import mock
from pyu2f import errors
from pyu2f import hardware
def testRegisterInvalidParams(self):
    mock_transport = mock.MagicMock()
    sk = hardware.SecurityKey(mock_transport)
    self.assertRaises(errors.InvalidRequestError, sk.CmdRegister, '1234', '1234')