import base64
import copy
from datetime import datetime
import json
import six
from six.moves import http_client
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import jwt
def from_credentials(self, target_credentials, target_audience=None):
    return self.__class__(target_credentials=target_credentials, target_audience=target_audience, include_email=self._include_email, quota_project_id=self._quota_project_id)