import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
def test_facebook_login(self):
    response = self.fetch('/facebook/client/login', follow_redirects=False)
    self.assertEqual(response.code, 302)
    self.assertTrue('/facebook/server/authorize?' in response.headers['Location'])
    response = self.fetch('/facebook/client/login?code=1234', follow_redirects=False)
    self.assertEqual(response.code, 200)
    user = json_decode(response.body)
    self.assertEqual(user['access_token'], 'asdf')
    self.assertEqual(user['session_expires'], '3600')