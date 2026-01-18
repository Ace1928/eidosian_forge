import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f import u2f
def testAuthenticateAllKeysInvalid(self):
    mock_sk = mock.MagicMock()
    mock_sk.CmdAuthenticate.side_effect = errors.InvalidKeyHandleError
    mock_sk.CmdVersion.return_value = b'U2F_V2'
    u2f_api = u2f.U2FInterface(mock_sk)
    with self.assertRaises(errors.U2FError) as cm:
        u2f_api.Authenticate('testapp', b'ABCD', [model.RegisteredKey('khA'), model.RegisteredKey('khB')])
    self.assertEquals(cm.exception.code, errors.U2FError.DEVICE_INELIGIBLE)
    u2f_api = u2f.U2FInterface(mock_sk)