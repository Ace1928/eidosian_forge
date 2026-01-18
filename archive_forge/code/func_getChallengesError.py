import base64
import json
import os
import unittest
import mock
from six.moves import http_client
from six.moves import urllib
from oauth2client import client
from google_reauth import challenges
from google_reauth import reauth
from google_reauth import errors
from google_reauth import reauth_creds
from google_reauth import _reauth_client
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
from pyu2f import model
from pyu2f import u2f
def getChallengesError(self, content):

    def side_effect(*args, **kwargs):
        uri = kwargs['uri'] if 'uri' in kwargs else args[0]
        if uri == self.oauth_api_url:
            return (_ok_response, json.dumps({'access_token': 'access_token_for_reauth'}))
        if uri == _reauth_client._REAUTH_API + ':start':
            return (None, content)
    with mock.patch('httplib2.Http.request', side_effect=side_effect) as request_mock:
        with self.assertRaises(errors.ReauthAPIError) as e:
            reauth.get_rapt_token(request_mock, self.client_id, self.client_secret, 'some_refresh_token', self.oauth_api_url)
        self.assertEqual(2, request_mock.call_count)