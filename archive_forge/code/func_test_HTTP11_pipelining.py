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
def test_HTTP11_pipelining(test_client):
    """Test HTTP/1.1 pipelining.

    :py:mod:`http.client` doesn't support this directly.
    """
    conn = test_client.get_connection()
    conn.putrequest('GET', '/hello', skip_host=True)
    conn.putheader('Host', conn.host)
    conn.endheaders()
    for trial in range(5):
        conn._output(('GET /hello?%s HTTP/1.1' % trial).encode('iso-8859-1'))
        conn._output(('Host: %s' % conn.host).encode('ascii'))
        conn._send_output()
        response = conn.response_class(conn.sock, method='GET')
        response.fp = conn.sock.makefile('rb', 0)
        response.begin()
        body = response.read(13)
        assert response.status == 200
        assert body == b'Hello, world!'
    response = conn.response_class(conn.sock, method='GET')
    response.begin()
    body = response.read()
    assert response.status == 200
    assert body == b'Hello, world!'
    conn.close()