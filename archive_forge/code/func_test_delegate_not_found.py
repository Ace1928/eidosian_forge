from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
def test_delegate_not_found(self):
    response = self.fetch('/404')
    self.assertEqual(response.code, 404)