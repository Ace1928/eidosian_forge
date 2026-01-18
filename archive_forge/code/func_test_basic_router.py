from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
def test_basic_router(self):
    response = self.fetch('/any_request')
    self.assertEqual(response.body, b'OK')