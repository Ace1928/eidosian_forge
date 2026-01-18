import datetime
import io
import json
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
from google.oauth2 import utils
def constructor_args(self):
    return {'audience': self._audience, 'refresh_token': self._refresh_token, 'token_url': self._token_url, 'token_info_url': self._token_info_url, 'client_id': self._client_id, 'client_secret': self._client_secret, 'token': self.token, 'expiry': self.expiry, 'revoke_url': self._revoke_url, 'scopes': self._scopes, 'quota_project_id': self._quota_project_id}