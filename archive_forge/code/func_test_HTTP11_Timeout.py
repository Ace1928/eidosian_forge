import errno
import socket
import time
import logging
import traceback as traceback_
from collections import namedtuple
import http.client
import urllib.request
import pytest
from jaraco.text import trim, unwrap
from cheroot.test import helper, webtest
from cheroot._compat import IS_CI, IS_MACOS, IS_PYPY, IS_WINDOWS
import cheroot.server
@pytest.mark.parametrize('timeout_before_headers', (True, False))
def test_HTTP11_Timeout(test_client, timeout_before_headers):
    """Check timeout without sending any data.

    The server will close the connection with a 408.
    """
    conn = test_client.get_connection()
    conn.auto_open = False
    conn.connect()
    if not timeout_before_headers:
        conn.send(b'GET /hello HTTP/1.1')
        conn.send(('Host: %s' % conn.host).encode('ascii'))
    time.sleep(timeout * 2)
    response = conn.response_class(conn.sock, method='GET')
    response.begin()
    assert response.status == 408
    conn.close()