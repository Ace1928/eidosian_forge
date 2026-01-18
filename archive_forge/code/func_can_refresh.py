import datetime
import io
import json
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
from google.oauth2 import utils
@property
def can_refresh(self):
    return all((self._refresh_token, self._token_url, self._client_id, self._client_secret))