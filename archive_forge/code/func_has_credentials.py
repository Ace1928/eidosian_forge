import importlib
import django.conf
from django.core import exceptions
from django.core import urlresolvers
from six.moves.urllib import parse
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
from oauth2client.contrib.django_util import storage
def has_credentials(self):
    """Returns True if there are valid credentials for the current user
        and required scopes."""
    credentials = _credentials_from_request(self.request)
    return credentials and (not credentials.invalid) and credentials.has_scopes(self._get_scopes())