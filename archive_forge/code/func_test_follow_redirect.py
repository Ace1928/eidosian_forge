import base64
import binascii
from contextlib import closing
import copy
import gzip
import threading
import datetime
from io import BytesIO
import subprocess
import sys
import time
import typing  # noqa: F401
import unicodedata
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen
from tornado.httpclient import (
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream
from tornado.log import gen_log, app_log
from tornado import netutil
from tornado.testing import AsyncHTTPTestCase, bind_unused_port, gen_test, ExpectLog
from tornado.test.util import skipOnTravis, ignore_deprecation
from tornado.web import Application, RequestHandler, url
from tornado.httputil import format_timestamp, HTTPHeaders
def test_follow_redirect(self):
    response = self.fetch('/countdown/2', follow_redirects=False)
    self.assertEqual(302, response.code)
    self.assertTrue(response.headers['Location'].endswith('/countdown/1'))
    response = self.fetch('/countdown/2')
    self.assertEqual(200, response.code)
    self.assertTrue(response.effective_url.endswith('/countdown/0'))
    self.assertEqual(b'Zero', response.body)