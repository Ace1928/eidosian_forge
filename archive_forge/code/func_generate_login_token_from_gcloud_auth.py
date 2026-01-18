from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import credentials
from google.auth import exceptions as google_auth_exceptions
from google.oauth2 import credentials as google_auth_creds
from googlecloudsdk.api_lib.auth import exceptions as auth_exceptions
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import google_auth_credentials as c_google_auth
from googlecloudsdk.core.credentials import store as c_store
import six
def generate_login_token_from_gcloud_auth(scopes):
    """Genearete a down-coped access token with given scopes for IAM DB authentication from gcloud credentials.

  Args:
    scopes: scopes to be included in the down-scoped token.

  Returns:
    Down-scoped access token.
  """
    cred = c_store.Load(allow_account_impersonation=True, use_google_auth=True, with_access_token_cache=False)
    cred = _downscope_credential(cred, scopes)
    c_store.Refresh(cred)
    if c_creds.IsOauth2ClientCredentials(cred):
        token = cred.access_token
    else:
        token = cred.token
    if not token:
        raise auth_exceptions.InvalidCredentialsError('No access token could be obtained from the current credentials.')
    return token