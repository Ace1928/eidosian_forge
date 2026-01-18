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
def test_destructor_log(self):
    proc = subprocess.run([sys.executable, '-c', 'from tornado.httpclient import HTTPClient; f = lambda: None; c = HTTPClient()'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, timeout=15)
    if proc.stdout:
        print('STDOUT:')
        print(to_unicode(proc.stdout))
    if proc.stdout:
        self.fail('subprocess produced unexpected output')