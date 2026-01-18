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
@pytest.mark.parametrize('target_scheme', ['http', 'https'])
def test_https_proxymanager_connected_to_http_proxy(self, target_scheme):
    if target_scheme == 'https' and sys.version_info[0] == 2:
        pytest.skip("HTTPS-in-HTTPS isn't supported on Python 2")
    errored = Event()

    def http_socket_handler(listener):
        sock = listener.accept()[0]
        sock.send(b'HTTP/1.0 501 Not Implemented\r\nConnection: close\r\n\r\n')
        errored.wait()
        sock.close()
    self._start_server(http_socket_handler)
    base_url = 'https://%s:%d' % (self.host, self.port)
    with ProxyManager(base_url, cert_reqs='NONE') as proxy:
        with pytest.raises(MaxRetryError) as e:
            proxy.request('GET', '%s://example.com' % target_scheme, retries=0)
        errored.set()
        assert type(e.value.reason) == ProxyError
        assert 'Your proxy appears to only use HTTP and not HTTPS' in str(e.value.reason)