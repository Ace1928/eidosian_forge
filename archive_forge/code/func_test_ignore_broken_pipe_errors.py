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
@notWindows
def test_ignore_broken_pipe_errors(self, monkeypatch):
    sock_shut = Event()
    orig_connect = HTTPConnection.connect
    buf = 'a' * 1024 * 1024 * 4

    def connect_and_wait(*args, **kw):
        ret = orig_connect(*args, **kw)
        assert sock_shut.wait(5)
        return ret

    def socket_handler(listener):
        for i in range(2):
            sock = listener.accept()[0]
            sock.send(b'HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 10\r\n\r\nxxxxxxxxxx')
            sock.shutdown(socket.SHUT_RDWR)
            sock_shut.set()
            sock.close()
    monkeypatch.setattr(HTTPConnection, 'connect', connect_and_wait)
    self._start_server(socket_handler)
    with HTTPConnectionPool(self.host, self.port) as pool:
        r = pool.request('POST', '/', body=buf)
        assert r.status == 404
        assert r.headers['content-length'] == '10'
        assert r.data == b'xxxxxxxxxx'
        r = pool.request('POST', '/admin', chunked=True, body=buf)
        assert r.status == 404
        assert r.headers['content-length'] == '10'
        assert r.data == b'xxxxxxxxxx'