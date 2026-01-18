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
def test_cookie_tampering_future_timestamp(self):
    handler = CookieTestRequestHandler()
    handler.set_signed_cookie('foo', binascii.a2b_hex(b'd76df8e7aefc'), version=1)
    cookie = handler._cookies['foo']
    match = re.match(b'12345678\\|([0-9]+)\\|([0-9a-f]+)', cookie)
    assert match is not None
    timestamp = match.group(1)
    sig = match.group(2)
    self.assertEqual(_create_signature_v1(handler.application.settings['cookie_secret'], 'foo', '12345678', timestamp), sig)
    self.assertEqual(_create_signature_v1(handler.application.settings['cookie_secret'], 'foo', '1234', b'5678' + timestamp), sig)
    handler._cookies['foo'] = utf8('1234|5678%s|%s' % (to_basestring(timestamp), to_basestring(sig)))
    with ExpectLog(gen_log, 'Cookie timestamp in future'):
        self.assertTrue(handler.get_signed_cookie('foo', min_version=1) is None)