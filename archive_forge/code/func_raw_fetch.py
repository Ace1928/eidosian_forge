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
def raw_fetch(self, headers, body, newline=b'\r\n'):
    with closing(IOStream(socket.socket())) as stream:
        self.io_loop.run_sync(lambda: stream.connect(('127.0.0.1', self.get_http_port())))
        stream.write(newline.join(headers + [utf8('Content-Length: %d' % len(body))]) + newline + newline + body)
        start_line, headers, body = self.io_loop.run_sync(lambda: read_stream_body(stream))
        return body