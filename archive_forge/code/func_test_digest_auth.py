from hashlib import md5
import unittest
from tornado.escape import utf8
from tornado.testing import AsyncHTTPTestCase
from tornado.test import httpclient_test
from tornado.web import Application, RequestHandler
def test_digest_auth(self):
    response = self.fetch('/digest', auth_mode='digest', auth_username='foo', auth_password='bar')
    self.assertEqual(response.body, b'ok')