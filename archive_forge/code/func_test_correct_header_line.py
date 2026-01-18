import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def test_correct_header_line(self):

    def request_handler(listener):
        sock = listener.accept()[0]
        handler = handle_socks4_negotiation(sock)
        addr, port = next(handler)
        assert addr == b'example.com'
        assert port == 80
        handler.send(True)
        buf = b''
        while True:
            buf += sock.recv(65535)
            if buf.endswith(b'\r\n\r\n'):
                break
        assert buf.startswith(b'GET / HTTP/1.1')
        assert b'Host: example.com' in buf
        sock.sendall(b'HTTP/1.1 200 OK\r\nServer: SocksTestServer\r\nContent-Length: 0\r\n\r\n')
        sock.close()
    self._start_server(request_handler)
    proxy_url = 'socks4a://%s:%s' % (self.host, self.port)
    with socks.SOCKSProxyManager(proxy_url) as pm:
        response = pm.request('GET', 'http://example.com')
        assert response.status == 200