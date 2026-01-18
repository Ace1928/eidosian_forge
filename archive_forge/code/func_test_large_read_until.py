from tornado.concurrent import Future
from tornado import gen
from tornado import netutil
from tornado.ioloop import IOLoop
from tornado.iostream import (
from tornado.httputil import HTTPHeaders
from tornado.locks import Condition, Event
from tornado.log import gen_log
from tornado.netutil import ssl_options_to_context, ssl_wrap_socket
from tornado.platform.asyncio import AddThreadSelectorEventLoop
from tornado.tcpserver import TCPServer
from tornado.testing import (
from tornado.test.util import (
from tornado.web import RequestHandler, Application
import asyncio
import errno
import hashlib
import logging
import os
import platform
import random
import socket
import ssl
import typing
from unittest import mock
import unittest
@gen_test
def test_large_read_until(self: typing.Any):
    rs, ws = (yield self.make_iostream_pair())
    try:
        if isinstance(rs, SSLIOStream) and platform.python_implementation() == 'PyPy':
            raise unittest.SkipTest('pypy gc causes problems with openssl')
        NUM_KB = 4096
        for i in range(NUM_KB):
            ws.write(b'A' * 1024)
        ws.write(b'\r\n')
        data = (yield rs.read_until(b'\r\n'))
        self.assertEqual(len(data), NUM_KB * 1024 + 2)
    finally:
        ws.close()
        rs.close()