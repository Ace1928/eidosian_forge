import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def test_connection_timeouts(self):
    event = threading.Event()

    def request_handler(listener):
        event.wait()
    self._start_server(request_handler)
    proxy_url = 'socks5h://%s:%s' % (self.host, self.port)
    with socks.SOCKSProxyManager(proxy_url) as pm:
        with pytest.raises(ConnectTimeoutError):
            pm.request('GET', 'http://example.com', timeout=SHORT_TIMEOUT, retries=False)
        event.set()