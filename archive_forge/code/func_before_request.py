import copy
import datetime
import json
import cachetools
import six
from six.moves import urllib
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import crypt
from google.auth import exceptions
import google.auth.credentials
def before_request(self, request, method, url, headers):
    """Performs credential-specific before request logic.

        Args:
            request (Any): Unused. JWT credentials do not need to make an
                HTTP request to refresh.
            method (str): The request's HTTP method.
            url (str): The request's URI. This is used as the audience claim
                when generating the JWT.
            headers (Mapping): The request's headers.
        """
    parts = urllib.parse.urlsplit(url)
    audience = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
    token = self._get_jwt_for_audience(audience)
    self.apply(headers, token=token)