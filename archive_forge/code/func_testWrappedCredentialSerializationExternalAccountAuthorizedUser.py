import datetime
import json
import httplib2
from google.auth import aws
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import pluggable
from gslib.tests import testcase
from gslib.utils.wrapped_credentials import WrappedCredentials
import oauth2client
from six import add_move, MovedModule
from six.moves import mock
def testWrappedCredentialSerializationExternalAccountAuthorizedUser(self):
    """Test logic for converting Wrapped Credentials to and from JSON for serialization."""
    creds = WrappedCredentials(external_account_authorized_user.Credentials(audience='//iam.googleapis.com/locations/global/workforcePools/$WORKFORCE_POOL_ID/providers/$PROVIDER_ID', refresh_token='refreshToken', token_url='https://sts.googleapis.com/v1/oauth/token', token_info_url='https://sts.googleapis.com/v1/instrospect', client_id='clientId', client_secret='clientSecret'))
    creds.access_token = ACCESS_TOKEN
    creds.token_expiry = datetime.datetime(2001, 12, 5, 0, 0)
    creds_json = creds.to_json()
    json_values = json.loads(creds_json)
    expected_json_values = {'_class': 'WrappedCredentials', '_module': 'gslib.utils.wrapped_credentials', 'client_id': 'clientId', 'access_token': ACCESS_TOKEN, 'token_expiry': '2001-12-05T00:00:00Z', 'client_secret': 'clientSecret', 'refresh_token': 'refreshToken', 'id_token': None, 'id_token_jwt': None, 'invalid': False, 'revoke_uri': None, 'scopes': [], 'token_info_uri': None, 'token_response': None, 'token_uri': None, 'user_agent': None, '_base': {'type': 'external_account_authorized_user', 'audience': '//iam.googleapis.com/locations/global/workforcePools/$WORKFORCE_POOL_ID/providers/$PROVIDER_ID', 'token': ACCESS_TOKEN, 'expiry': '2001-12-05T00:00:00Z', 'token_url': 'https://sts.googleapis.com/v1/oauth/token', 'token_info_url': 'https://sts.googleapis.com/v1/instrospect', 'refresh_token': 'refreshToken', 'client_id': 'clientId', 'client_secret': 'clientSecret'}}
    self.assertEqual(json_values, expected_json_values)
    creds2 = WrappedCredentials.from_json(creds_json)
    self.assertIsInstance(creds2, WrappedCredentials)
    self.assertIsInstance(creds2._base, external_account_authorized_user.Credentials)
    self.assertEqual(creds2.client_id, 'clientId')