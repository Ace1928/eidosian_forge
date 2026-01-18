from contextlib import closing
import getpass
import os
import socket
import unittest
from tornado.concurrent import Future
from tornado.netutil import bind_sockets, Resolver
from tornado.queues import Queue
from tornado.tcpclient import TCPClient, _Connector
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, gen_test
from tornado.test.util import skipIfNoIPv6, refusing_port, skipIfNonUnix
from tornado.gen import TimeoutError
import typing
@gen_test
def do_test_connect(self, family, host, source_ip=None, source_port=None):
    port = self.start_server(family)
    stream = (yield self.client.connect(host, port, source_ip=source_ip, source_port=source_port, af=family))
    assert self.server is not None
    server_stream = (yield self.server.queue.get())
    with closing(stream):
        stream.write(b'hello')
        data = (yield server_stream.read_bytes(5))
        self.assertEqual(data, b'hello')