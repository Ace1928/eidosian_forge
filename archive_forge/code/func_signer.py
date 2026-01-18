import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import iam
from google.auth import jwt
from google.auth.compute_engine import _metadata
from google.oauth2 import _client
@property
@_helpers.copy_docstring(credentials.Signing)
def signer(self):
    return self._signer