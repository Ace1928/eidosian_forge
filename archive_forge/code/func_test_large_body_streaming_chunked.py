from tornado import gen, netutil
from tornado.escape import (
from tornado.http1connection import HTTP1Connection
from tornado.httpclient import HTTPError
from tornado.httpserver import HTTPServer
from tornado.httputil import (
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import ssl_options_to_context
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import (
from tornado.test.util import skipOnTravis
from tornado.web import Application, RequestHandler, stream_request_body
from contextlib import closing
import datetime
import gzip
import logging
import os
import shutil
import socket
import ssl
import sys
import tempfile
import textwrap
import unittest
import urllib.parse
from io import BytesIO
import typing
@unittest.skipIf(os.name == 'nt', 'flaky on windows')
def test_large_body_streaming_chunked(self):
    with ExpectLog(gen_log, '.*chunked body too large', level=logging.INFO):
        response = self.fetch('/streaming', method='PUT', body_producer=lambda write: write(b'a' * 10240))
    self.assertEqual(response.code, 400)