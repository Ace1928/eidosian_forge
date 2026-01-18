from dummyserver.server import (
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, ProxyManager, util
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import HTTPConnection, _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.poolmanager import proxy_from_url
from urllib3.util import ssl_, ssl_wrap_socket
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout
from .. import LogRecorder, has_alpn, onlyPy3
import os
import os.path
import select
import shutil
import socket
import ssl
import sys
import tempfile
from collections import OrderedDict
from test import (
from threading import Event
import mock
import pytest
import trustme
def test_connect_reconn(self):

    def proxy_ssl_one(listener):
        sock = listener.accept()[0]
        buf = b''
        while not buf.endswith(b'\r\n\r\n'):
            buf += sock.recv(65536)
        s = buf.decode('utf-8')
        if not s.startswith('CONNECT '):
            sock.send('HTTP/1.1 405 Method not allowed\r\nAllow: CONNECT\r\n\r\n'.encode('utf-8'))
            sock.close()
            return
        if not s.startswith('CONNECT %s:443' % (self.host,)):
            sock.send('HTTP/1.1 403 Forbidden\r\n\r\n'.encode('utf-8'))
            sock.close()
            return
        sock.send('HTTP/1.1 200 Connection Established\r\n\r\n'.encode('utf-8'))
        ssl_sock = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
        buf = b''
        while not buf.endswith(b'\r\n\r\n'):
            buf += ssl_sock.recv(65536)
        ssl_sock.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nHi'.encode('utf-8'))
        ssl_sock.close()

    def echo_socket_handler(listener):
        proxy_ssl_one(listener)
        proxy_ssl_one(listener)
    self._start_server(echo_socket_handler)
    base_url = 'http://%s:%d' % (self.host, self.port)
    with proxy_from_url(base_url, ca_certs=DEFAULT_CA) as proxy:
        url = 'https://{0}'.format(self.host)
        conn = proxy.connection_from_url(url)
        r = conn.urlopen('GET', url, retries=0)
        assert r.status == 200
        r = conn.urlopen('GET', url, retries=0)
        assert r.status == 200