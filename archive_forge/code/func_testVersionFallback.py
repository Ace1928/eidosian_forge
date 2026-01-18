import sys
import mock
from pyu2f import errors
from pyu2f import hardware
def testVersionFallback(self):
    mock_transport = mock.MagicMock()
    sk = hardware.SecurityKey(mock_transport)
    mock_transport.SendMsgBytes.side_effect = [bytearray([103, 0]), bytearray(b'U2F_V2\x90\x00')]
    reply = sk.CmdVersion()
    self.assertEquals(reply, bytearray(b'U2F_V2'))
    self.assertEquals(mock_transport.SendMsgBytes.call_count, 2)
    (sent_msg,), _ = mock_transport.SendMsgBytes.call_args_list[0]
    self.assertEquals(len(sent_msg), 7)
    self.assertEquals(sent_msg[0:4], bytearray([0, 3, 0, 0]))
    self.assertEquals(sent_msg[4:7], bytearray([0, 0, 0]))
    (sent_msg,), _ = mock_transport.SendMsgBytes.call_args_list[1]
    self.assertEquals(len(sent_msg), 9)
    self.assertEquals(sent_msg[0:4], bytearray([0, 3, 0, 0]))
    self.assertEquals(sent_msg[4:7], bytearray([0, 0, 0]))
    self.assertEquals(sent_msg[7:9], bytearray([0, 0]))