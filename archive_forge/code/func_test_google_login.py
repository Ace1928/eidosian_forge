import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
def test_google_login(self):
    response = self.fetch('/client/login')
    self.assertDictEqual({'name': 'Foo', 'email': 'foo@example.com', 'access_token': 'fake-access-token'}, json_decode(response.body))