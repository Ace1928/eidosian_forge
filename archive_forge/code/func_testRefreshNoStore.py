import base64
import datetime
import json
import os
import unittest
import mock
from mock import patch
from six.moves import http_client
from six.moves import urllib
from oauth2client import client
from oauth2client import client
from google_reauth import reauth
from google_reauth import errors
from google_reauth import reauth_creds
from google_reauth import _reauth_client
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
def testRefreshNoStore(self):

    def request_side_effect(self, *args, **kwargs):
        return _token_response
    creds = self._get_creds()
    creds._do_refresh_request(self._http_mock(request_side_effect))
    self._check_credentials(creds, None, 'new_access_token', 'new_refresh_token', datetime.datetime(2018, 3, 2, 22, 26, 13), False)