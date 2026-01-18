import sys
import mock
from pyu2f import errors
from pyu2f import hardware
def testVersionErrors(self):
    mock_transport = mock.MagicMock()
    sk = hardware.SecurityKey(mock_transport)
    mock_transport.SendMsgBytes.return_value = bytearray([250, 5])
    self.assertRaises(errors.ApduError, sk.CmdVersion)
    self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)