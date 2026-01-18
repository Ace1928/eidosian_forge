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
@skipIfNoIPv6
def test_connect_ipv6_ipv6(self):
    self.skipIfLocalhostV4()
    self.do_test_connect(socket.AF_INET6, '::1')