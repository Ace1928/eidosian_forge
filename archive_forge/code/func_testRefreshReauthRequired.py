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
def testRefreshReauthRequired(self):
    responses = [_token_response, (_error_response, json.dumps({'error': 'invalid_grant', 'error_subtype': 'rapt_required'}))]

    def request_side_effect(self, *args, **kwargs):
        return responses.pop()
    self._run_refresh_test(self._http_mock(request_side_effect), 'new_access_token', 'new_refresh_token', datetime.datetime(2018, 3, 2, 22, 26, 13), False)