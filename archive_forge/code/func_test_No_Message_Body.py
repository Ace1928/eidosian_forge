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
def test_No_Message_Body(test_client):
    """Test HTTP queries with an empty response body."""
    http_connection = test_client.get_connection()
    http_connection.auto_open = False
    http_connection.connect()
    status_line, actual_headers, actual_resp_body = test_client.get('/pov', http_conn=http_connection)
    actual_status = int(status_line[:3])
    assert actual_status == 200
    assert status_line[4:] == 'OK'
    assert actual_resp_body == pov.encode()
    assert not header_exists('Connection', actual_headers)
    status_line, actual_headers, actual_resp_body = test_client.get('/custom/204', http_conn=http_connection)
    actual_status = int(status_line[:3])
    assert actual_status == 204
    assert not header_exists('Content-Length', actual_headers)
    assert actual_resp_body == b''
    assert not header_exists('Connection', actual_headers)
    status_line, actual_headers, actual_resp_body = test_client.get('/custom/304', http_conn=http_connection)
    actual_status = int(status_line[:3])
    assert actual_status == 304
    assert not header_exists('Content-Length', actual_headers)
    assert actual_resp_body == b''
    assert not header_exists('Connection', actual_headers)
    http_connection.close()