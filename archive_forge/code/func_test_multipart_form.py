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
def test_multipart_form(self):
    response = self.raw_fetch([b'POST /multipart HTTP/1.0', b'Content-Type: multipart/form-data; boundary=1234567890', b'X-Header-encoding-test: \xe9'], b'\r\n'.join([b'Content-Disposition: form-data; name=argument', b'', 'á'.encode('utf-8'), b'--1234567890', 'Content-Disposition: form-data; name="files"; filename="ó"'.encode('utf8'), b'', 'ú'.encode('utf-8'), b'--1234567890--', b'']))
    data = json_decode(response)
    self.assertEqual('é', data['header'])
    self.assertEqual('á', data['argument'])
    self.assertEqual('ó', data['filename'])
    self.assertEqual('ú', data['filebody'])