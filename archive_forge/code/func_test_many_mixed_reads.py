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
def test_many_mixed_reads(self):
    r = random.Random(42)
    nbytes = 1000000
    rs, ws = (yield self.make_iostream_pair())
    produce_hash = hashlib.sha1()
    consume_hash = hashlib.sha1()

    @gen.coroutine
    def produce():
        remaining = nbytes
        while remaining > 0:
            size = r.randint(1, min(1000, remaining))
            data = os.urandom(size)
            produce_hash.update(data)
            yield ws.write(data)
            remaining -= size
        assert remaining == 0

    @gen.coroutine
    def consume():
        remaining = nbytes
        while remaining > 0:
            if r.random() > 0.5:
                size = r.randint(1, min(1000, remaining))
                data = (yield rs.read_bytes(size))
                consume_hash.update(data)
                remaining -= size
            else:
                size = r.randint(1, min(1000, remaining))
                buf = bytearray(size)
                n = (yield rs.read_into(buf))
                assert n == size
                consume_hash.update(buf)
                remaining -= size
        assert remaining == 0
    try:
        yield [produce(), consume()]
        assert produce_hash.hexdigest() == consume_hash.hexdigest()
    finally:
        ws.close()
        rs.close()