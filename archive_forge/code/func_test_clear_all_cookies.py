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
def test_clear_all_cookies(self):
    response = self.fetch('/', headers={'Cookie': 'foo=bar; baz=xyzzy'})
    set_cookies = sorted(response.headers.get_list('Set-Cookie'))
    self.assertTrue(set_cookies[0].startswith('baz=;') or set_cookies[0].startswith('baz="";'))
    self.assertTrue(set_cookies[1].startswith('foo=;') or set_cookies[1].startswith('foo="";'))