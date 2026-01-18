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
@gen_test
def test_body_size_override_reset(self):
    stream = IOStream(socket.socket())
    try:
        yield stream.connect(('127.0.0.1', self.get_http_port()))
        stream.write(b'PUT /streaming?expected_size=10240 HTTP/1.1\r\nContent-Length: 10240\r\n\r\n')
        stream.write(b'a' * 10240)
        start_line, headers, response = (yield read_stream_body(stream))
        self.assertEqual(response, b'10240')
        stream.write(b'PUT /streaming HTTP/1.1\r\nContent-Length: 10240\r\n\r\n')
        with ExpectLog(gen_log, '.*Content-Length too long', level=logging.INFO):
            data = (yield stream.read_until_close())
        self.assertEqual(data, b'HTTP/1.1 400 Bad Request\r\n\r\n')
    finally:
        stream.close()