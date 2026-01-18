import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f import u2f
def testRegisterSuccessWithPreviousKeys(self):
    mock_sk = mock.MagicMock()
    mock_sk.CmdAuthenticate.side_effect = errors.InvalidKeyHandleError
    mock_sk.CmdRegister.side_effect = [errors.TUPRequiredError, 'regdata']
    mock_sk.CmdVersion.return_value = b'U2F_V2'
    u2f_api = u2f.U2FInterface(mock_sk)
    resp = u2f_api.Register('testapp', b'ABCD', [model.RegisteredKey('khA')])
    self.assertEquals(mock_sk.CmdAuthenticate.call_count, 1)
    self.assertTrue(mock_sk.CmdAuthenticate.call_args[0][3])
    self.assertEquals(mock_sk.CmdRegister.call_count, 2)
    self.assertEquals(mock_sk.CmdWink.call_count, 1)
    self.assertEquals(resp.client_data.raw_server_challenge, b'ABCD')
    self.assertEquals(resp.client_data.typ, 'navigator.id.finishEnrollment')
    self.assertEquals(resp.registration_data, 'regdata')