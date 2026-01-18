import socket
import subprocess
import sys
import textwrap
import unittest
from tornado import gen
from tornado.iostream import IOStream
from tornado.log import app_log
from tornado.tcpserver import TCPServer
from tornado.test.util import skipIfNonUnix
from tornado.testing import AsyncTestCase, ExpectLog, bind_unused_port, gen_test
from typing import Tuple
@gen_test
def test_handle_stream_native_coroutine(self):

    class TestServer(TCPServer):

        async def handle_stream(self, stream, address):
            stream.write(b'data')
            stream.close()
    sock, port = bind_unused_port()
    server = TestServer()
    server.add_socket(sock)
    client = IOStream(socket.socket())
    yield client.connect(('localhost', port))
    result = (yield client.read_until_close())
    self.assertEqual(result, b'data')
    server.stop()
    client.close()