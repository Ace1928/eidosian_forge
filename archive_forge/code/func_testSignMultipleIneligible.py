import base64
import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import localauthenticator
@mock.patch.object(localauthenticator.u2f, 'GetLocalU2FInterface')
def testSignMultipleIneligible(self, mock_get_u2f_method):
    """Test signing with multiple keys registered, but none eligible."""
    mock_u2f = mock.MagicMock()
    mock_get_u2f_method.return_value = mock_u2f
    mock_authenticate = mock.MagicMock()
    mock_u2f.Authenticate = mock_authenticate
    mock_authenticate.side_effect = errors.U2FError(errors.U2FError.DEVICE_INELIGIBLE)
    challenge_item = {'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}
    challenge_data = [challenge_item, challenge_item]
    authenticator = localauthenticator.LocalAuthenticator('testorigin')
    with self.assertRaises(errors.U2FError) as cm:
        authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)
    self.assertEquals(cm.exception.code, errors.U2FError.DEVICE_INELIGIBLE)