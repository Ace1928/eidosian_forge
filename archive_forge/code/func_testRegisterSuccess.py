import sys
import mock
from pyu2f import errors
from pyu2f import hardware
def testRegisterSuccess(self):
    mock_transport = mock.MagicMock()
    sk = hardware.SecurityKey(mock_transport)
    challenge_param = b'01234567890123456789012345678901'
    app_param = b'01234567890123456789012345678901'
    mock_transport.SendMsgBytes.return_value = bytearray([1, 2, 144, 0])
    reply = sk.CmdRegister(challenge_param, app_param)
    self.assertEquals(reply, bytearray([1, 2]))
    self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)
    (sent_msg,), _ = mock_transport.SendMsgBytes.call_args
    self.assertEquals(sent_msg[0:4], bytearray([0, 1, 3, 0]))
    self.assertEquals(sent_msg[7:-2], bytearray(challenge_param + app_param))