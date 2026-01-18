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
@pytest.mark.parametrize('invalid_terminator', (b'\n\n', b'\r\n\n'))
def test_No_CRLF(test_client, invalid_terminator):
    """Test HTTP queries with no valid CRLF terminators."""
    conn = test_client.get_connection()
    conn.send(b'GET /hello HTTP/1.1%s' % invalid_terminator)
    response = conn.response_class(conn.sock, method='GET')
    response.begin()
    actual_resp_body = response.read()
    expected_resp_body = b'HTTP requires CRLF terminators'
    assert actual_resp_body == expected_resp_body
    conn.close()