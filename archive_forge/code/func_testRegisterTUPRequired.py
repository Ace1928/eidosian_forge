import sys
import mock
from pyu2f import errors
from pyu2f import hardware
def testRegisterTUPRequired(self):
    mock_transport = mock.MagicMock()
    sk = hardware.SecurityKey(mock_transport)
    challenge_param = b'01234567890123456789012345678901'
    app_param = b'01234567890123456789012345678901'
    mock_transport.SendMsgBytes.return_value = bytearray([105, 133])
    self.assertRaises(errors.TUPRequiredError, sk.CmdRegister, challenge_param, app_param)
    self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)