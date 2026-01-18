import sys
import mock
from pyu2f import errors
from pyu2f import hardware
def testVersion(self):
    mock_transport = mock.MagicMock()
    sk = hardware.SecurityKey(mock_transport)
    mock_transport.SendMsgBytes.return_value = bytearray(b'U2F_V2\x90\x00')
    reply = sk.CmdVersion()
    self.assertEquals(reply, bytearray(b'U2F_V2'))
    self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)
    (sent_msg,), _ = mock_transport.SendMsgBytes.call_args
    self.assertEquals(sent_msg, bytearray([0, 3, 0, 0, 0, 0, 0]))