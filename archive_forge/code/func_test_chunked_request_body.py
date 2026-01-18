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
def test_chunked_request_body(self):
    self.stream.write(b'POST /echo HTTP/1.1\nTransfer-Encoding: chunked\nContent-Type: application/x-www-form-urlencoded\n\n4\nfoo=\n3\nbar\n0\n\n'.replace(b'\n', b'\r\n'))
    start_line, headers, response = self.io_loop.run_sync(lambda: read_stream_body(self.stream))
    self.assertEqual(json_decode(response), {'foo': ['bar']})