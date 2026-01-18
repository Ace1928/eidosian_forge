import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f import u2f
def testAuthenticateSuccessSkipInvalidVersion(self):
    mock_sk = mock.MagicMock()
    mock_sk.CmdAuthenticate.return_value = 'signature'
    mock_sk.CmdVersion.return_value = b'U2F_V2'
    u2f_api = u2f.U2FInterface(mock_sk)
    resp = u2f_api.Authenticate('testapp', b'ABCD', [model.RegisteredKey('khA', version='U2F_V3'), model.RegisteredKey('khB')])
    self.assertEquals(mock_sk.CmdAuthenticate.call_count, 1)
    self.assertEquals(mock_sk.CmdWink.call_count, 0)
    self.assertEquals(resp.key_handle, 'khB')
    self.assertEquals(resp.client_data.raw_server_challenge, b'ABCD')
    self.assertEquals(resp.client_data.typ, 'navigator.id.getAssertion')
    self.assertEquals(resp.signature_data, 'signature')