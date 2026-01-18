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
def test_closing_response_actually_closes_connection(self):
    done_closing = Event()
    complete = Event()

    def socket_handler(listener):
        sock = listener.accept()[0]
        buf = b''
        while not buf.endswith(b'\r\n\r\n'):
            buf = sock.recv(65536)
        sock.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n\r\n'.encode('utf-8'))
        done_closing.wait(timeout=LONG_TIMEOUT)
        sock.settimeout(LONG_TIMEOUT)
        new_data = sock.recv(65536)
        assert not new_data
        sock.close()
        complete.set()
    self._start_server(socket_handler)
    with HTTPConnectionPool(self.host, self.port) as pool:
        response = pool.request('GET', '/', retries=0, preload_content=False)
        assert response.status == 200
        response.close()
        done_closing.set()
        successful = complete.wait(timeout=LONG_TIMEOUT)
        assert successful, 'Timed out waiting for connection close'