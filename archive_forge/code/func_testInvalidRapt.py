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
def testInvalidRapt(self):
    responses = [(_error_response, json.dumps({'error': 'invalid_grant', 'error_subtype': 'rapt_required'})), (_error_response, json.dumps({'error': 'invalid_grant', 'error_subtype': 'rapt_required'}))]

    def request_side_effect(self, *args, **kwargs):
        return responses.pop()
    creds = self._get_creds()
    store = MockStore()
    creds.set_store(store)
    with self.assertRaises(client.HttpAccessTokenRefreshError):
        creds._do_refresh_request(self._http_mock(request_side_effect))
    self._check_credentials(creds, store, 'old_token', 'old_refresh_token', datetime.datetime(2018, 3, 2, 21, 26, 13), True)