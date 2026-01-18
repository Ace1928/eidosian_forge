from tornado.concurrent import Future
from tornado import gen
from tornado.escape import (
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import (
import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing  # noqa: F401
import unittest
import urllib.parse
def test_flow_control_compressed_body(self: typing.Any):
    bytesio = BytesIO()
    gzip_file = gzip.GzipFile(mode='w', fileobj=bytesio)
    gzip_file.write(b'abcdefghijklmnopqrstuvwxyz')
    gzip_file.close()
    compressed_body = bytesio.getvalue()
    response = self.fetch('/', body=compressed_body, method='POST', headers={'Content-Encoding': 'gzip'})
    response.rethrow()
    self.assertEqual(json_decode(response.body), dict(methods=['prepare', 'data_received', 'data_received', 'data_received', 'post']))