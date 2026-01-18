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
def testFromJsonAWSCredentials(self):
    creds = WrappedCredentials.from_json(json.dumps({'_base': {'audience': '//iam.googleapis.com/projects/123456/locations/global/workloadIdentityPools/POOL_ID/providers/PROVIDER_ID', 'credential_source': {'environment_id': 'aws1', 'region_url': 'http://169.254.169.254/latest/meta-data/placement/availability-zone', 'regional_cred_verification_url': 'https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15', 'url': 'http://169.254.169.254/latest/meta-data/iam/security-credentials'}, 'service_account_impersonation_url': 'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/service-1234@service-name.iam.gserviceaccount.com:generateAccessToken', 'subject_token_type': 'urn:ietf:params:aws:token-type:aws4_request', 'token_url': 'https://sts.googleapis.com/v1/token', 'type': 'external_account'}}))
    self.assertIsInstance(creds, WrappedCredentials)
    self.assertIsInstance(creds._base, external_account.Credentials)
    self.assertIsInstance(creds._base, aws.Credentials)