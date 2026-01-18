import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test__get_challenges_with_scopes():
    with mock.patch('google.oauth2._client._token_endpoint_request') as mock_token_endpoint_request:
        reauth._get_challenges(MOCK_REQUEST, ['SAML'], 'token', requested_scopes=['scope'])
        mock_token_endpoint_request.assert_called_with(MOCK_REQUEST, reauth._REAUTH_API + ':start', {'supportedChallengeTypes': ['SAML'], 'oauthScopesForDomainPolicyLookup': ['scope']}, access_token='token', use_json=True)