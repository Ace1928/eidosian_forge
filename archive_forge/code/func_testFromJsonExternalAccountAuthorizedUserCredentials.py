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
def testFromJsonExternalAccountAuthorizedUserCredentials(self):
    creds = WrappedCredentials.from_json(json.dumps({'_base': {'type': 'external_account_authorized_user', 'audience': '//iam.googleapis.com/locations/global/workforcePools/$WORKFORCE_POOL_ID/providers/$PROVIDER_ID', 'refresh_token': 'refreshToken', 'token_url': 'https://sts.googleapis.com/v1/oauth/token', 'token_info_url': 'https://sts.googleapis.com/v1/instrospect', 'client_id': 'clientId', 'client_secret': 'clientSecret'}}))
    self.assertIsInstance(creds, WrappedCredentials)
    self.assertIsInstance(creds._base, external_account_authorized_user.Credentials)