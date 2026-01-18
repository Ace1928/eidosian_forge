import importlib
import django.conf
from django.core import exceptions
from django.core import urlresolvers
from six.moves.urllib import parse
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
from oauth2client.contrib.django_util import storage
def get_authorize_redirect(self):
    """Creates a URl to start the OAuth2 authorization flow."""
    get_params = {'return_url': self.return_url, 'scopes': self._get_scopes()}
    return _redirect_with_params('google_oauth:authorize', **get_params)