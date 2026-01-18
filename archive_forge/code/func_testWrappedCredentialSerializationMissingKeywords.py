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
def testWrappedCredentialSerializationMissingKeywords(self):
    """Test logic for creating a Wrapped Credentials using keywords that exist in IdentityPool but not AWS."""
    creds = WrappedCredentials.from_json(json.dumps({'client_id': 'foo', 'access_token': ACCESS_TOKEN, 'token_expiry': '2001-12-05T00:00:00Z', '_base': {'type': 'external_account', 'audience': 'foo', 'subject_token_type': 'bar', 'token_url': 'https://sts.googleapis.com', 'credential_source': {'url': 'google.com', 'workforce_pool_user_project': '1234567890'}}}))
    self.assertIsInstance(creds, WrappedCredentials)
    self.assertIsInstance(creds._base, identity_pool.Credentials)