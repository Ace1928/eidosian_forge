import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def test_local_dns(self):

    def request_handler(listener):
        sock = listener.accept()[0]
        handler = handle_socks4_negotiation(sock)
        addr, port = next(handler)
        assert addr == '127.0.0.1'
        assert port == 80
        handler.send(True)
        while True:
            buf = sock.recv(65535)
            if buf.endswith(b'\r\n\r\n'):
                break
        sock.sendall(b'HTTP/1.1 200 OK\r\nServer: SocksTestServer\r\nContent-Length: 0\r\n\r\n')
        sock.close()
    self._start_server(request_handler)
    proxy_url = 'socks4://%s:%s' % (self.host, self.port)
    with socks.SOCKSProxyManager(proxy_url) as pm:
        response = pm.request('GET', 'http://localhost')
        assert response.status == 200
        assert response.headers['Server'] == 'SocksTestServer'
        assert response.data == b''