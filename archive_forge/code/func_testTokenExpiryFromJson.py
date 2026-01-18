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
def testTokenExpiryFromJson(self):
    cred = Oauth2WithReauthCredentials.from_json(json.dumps({'access_token': 'access_token', 'client_id': 'client_id', 'client_secret': 'client_secret', 'refresh_token': 'refresh_token', 'token_expiry': 'foo', 'token_uri': 'token_uri', 'user_agent': 'user_agent', 'invalid': False}))
    self.assertEqual(None, cred.token_expiry)
    cred = Oauth2WithReauthCredentials.from_json(json.dumps({'access_token': 'access_token', 'client_id': 'client_id', 'client_secret': 'client_secret', 'refresh_token': 'refresh_token', 'token_expiry': '2018-03-02T21:26:13Z', 'token_uri': 'token_uri', 'user_agent': 'user_agent', 'invalid': False}))
    self.assertEqual(datetime.datetime(2018, 3, 2, 21, 26, 13), cred.token_expiry)