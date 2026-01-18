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
def test_keepalive_conn_management(test_client):
    """Test management of Keep-Alive connections."""
    test_client.server_instance.timeout = 2

    def connection():
        http_connection = test_client.get_connection()
        http_connection.auto_open = False
        http_connection.connect()
        return http_connection

    def request(conn, keepalive=True):
        status_line, actual_headers, actual_resp_body = test_client.get('/page3', headers=[('Connection', 'Keep-Alive')], http_conn=conn, protocol='HTTP/1.0')
        actual_status = int(status_line[:3])
        assert actual_status == 200
        assert status_line[4:] == 'OK'
        assert actual_resp_body == pov.encode()
        if keepalive:
            assert header_has_value('Connection', 'Keep-Alive', actual_headers)
            assert header_has_value('Keep-Alive', 'timeout={test_client.server_instance.timeout}'.format(**locals()), actual_headers)
        else:
            assert not header_exists('Connection', actual_headers)
            assert not header_exists('Keep-Alive', actual_headers)

    def check_server_idle_conn_count(count, timeout=1.0):
        deadline = time.time() + timeout
        while True:
            n = test_client.server_instance._connections._num_connections
            if n == count:
                return
            assert time.time() <= deadline, ('idle conn count mismatch, wanted {count}, got {n}'.format(**locals()),)
    disconnect_errors = (http.client.BadStatusLine, http.client.CannotSendRequest, http.client.NotConnected)
    c1 = connection()
    request(c1)
    check_server_idle_conn_count(1)
    c2 = connection()
    request(c2)
    check_server_idle_conn_count(2)
    request(c1)
    check_server_idle_conn_count(2)
    c3 = connection()
    request(c3, keepalive=False)
    check_server_idle_conn_count(2)
    with pytest.raises(disconnect_errors):
        request(c3)
    check_server_idle_conn_count(2)
    time.sleep(1.2)
    request(c2)
    check_server_idle_conn_count(2)
    time.sleep(1.2)
    check_server_idle_conn_count(1)
    with pytest.raises(disconnect_errors):
        request(c1)
    check_server_idle_conn_count(1)
    request(c2)
    check_server_idle_conn_count(1)
    test_client.server_instance.timeout = timeout
    c1.close()
    c2.close()
    c3.close()