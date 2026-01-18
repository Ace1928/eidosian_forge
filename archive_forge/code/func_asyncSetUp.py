import socket
import typing  # noqa(F401)
from tornado.http1connection import HTTP1Connection
from tornado.httputil import HTTPMessageDelegate
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.netutil import add_accept_handler
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test
@gen_test
def asyncSetUp(self):
    listener, port = bind_unused_port()
    event = Event()

    def accept_callback(conn, addr):
        self.server_stream = IOStream(conn)
        self.addCleanup(self.server_stream.close)
        event.set()
    add_accept_handler(listener, accept_callback)
    self.client_stream = IOStream(socket.socket())
    self.addCleanup(self.client_stream.close)
    yield [self.client_stream.connect(('127.0.0.1', port)), event.wait()]
    self.io_loop.remove_handler(listener)
    listener.close()