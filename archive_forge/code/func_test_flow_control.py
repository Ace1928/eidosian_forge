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
def test_flow_control(self):
    MB = 1024 * 1024
    rs, ws = (yield self.make_iostream_pair(max_buffer_size=5 * MB))
    try:
        ws.write(b'a' * 10 * MB)
        yield rs.read_bytes(MB)
        yield gen.sleep(0.1)
        for i in range(9):
            yield rs.read_bytes(MB)
    finally:
        rs.close()
        ws.close()