from concurrent import futures
import logging
import re
import socket
import typing
import unittest
from tornado.concurrent import (
from tornado.escape import utf8, to_unicode
from tornado import gen
from tornado.iostream import IOStream
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test
def test_future(self: typing.Any):
    future = self.client.capitalize('hello')
    self.io_loop.add_future(future, self.stop)
    self.wait()
    self.assertEqual(future.result(), 'HELLO')