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
def test_100_Continue(test_client):
    """Test 100-continue header processing."""
    conn = test_client.get_connection()
    conn.putrequest('POST', '/upload', skip_host=True)
    conn.putheader('Host', conn.host)
    conn.putheader('Content-Type', 'text/plain')
    conn.putheader('Content-Length', '4')
    conn.endheaders()
    conn.send(b"d'oh")
    response = conn.response_class(conn.sock, method='POST')
    _version, status, _reason = response._read_status()
    assert status != 100
    conn.close()
    conn.connect()
    conn.putrequest('POST', '/upload', skip_host=True)
    conn.putheader('Host', conn.host)
    conn.putheader('Content-Type', 'text/plain')
    conn.putheader('Content-Length', '17')
    conn.putheader('Expect', '100-continue')
    conn.endheaders()
    response = conn.response_class(conn.sock, method='POST')
    version, status, reason = response._read_status()
    assert status == 100
    while True:
        line = response.fp.readline().strip()
        if line:
            pytest.fail('100 Continue should not output any headers. Got %r' % line)
        else:
            break
    body = b'I am a small file'
    conn.send(body)
    response.begin()
    status_line, _actual_headers, actual_resp_body = webtest.shb(response)
    actual_status = int(status_line[:3])
    assert actual_status == 200
    expected_resp_body = ("thanks for '%s'" % body).encode()
    assert actual_resp_body == expected_resp_body
    conn.close()