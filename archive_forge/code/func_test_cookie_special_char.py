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
def test_cookie_special_char(self):
    response = self.fetch('/special_char')
    headers = sorted(response.headers.get_list('Set-Cookie'))
    self.assertEqual(len(headers), 3)
    self.assertEqual(headers[0], 'equals="a=b"; Path=/')
    self.assertEqual(headers[1], 'quote="a\\"b"; Path=/')
    self.assertTrue(headers[2] in ('semicolon="a;b"; Path=/', 'semicolon="a\\073b"; Path=/'), headers[2])
    data = [('foo=a=b', 'a=b'), ('foo="a=b"', 'a=b'), ('foo="a;b"', '"a'), ('foo=a\\073b', 'a\\073b'), ('foo="a\\073b"', 'a;b'), ('foo="a\\"b"', 'a"b')]
    for header, expected in data:
        logging.debug('trying %r', header)
        response = self.fetch('/get', headers={'Cookie': header})
        self.assertEqual(response.body, utf8(expected))