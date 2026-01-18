import socket
import typing  # noqa(F401)
from tornado.http1connection import HTTP1Connection
from tornado.httputil import HTTPMessageDelegate
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.netutil import add_accept_handler
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test
@gen_test
def test_http10_no_content_length(self):
    conn = HTTP1Connection(self.client_stream, True)
    self.server_stream.write(b'HTTP/1.0 200 Not Modified\r\n\r\nhello')
    self.server_stream.close()
    event = Event()
    test = self
    body = []

    class Delegate(HTTPMessageDelegate):

        def headers_received(self, start_line, headers):
            test.code = start_line.code

        def data_received(self, data):
            body.append(data)

        def finish(self):
            event.set()
    yield conn.read_response(Delegate())
    yield event.wait()
    self.assertEqual(self.code, 200)
    self.assertEqual(b''.join(body), b'hello')