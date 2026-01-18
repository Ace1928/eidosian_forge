import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_garbage_in(test_client):
    """Test that server sends an error for garbage received over TCP."""
    c = test_client.get_connection()
    c._output(b'gjkgjklsgjklsgjkljklsg')
    c._send_output()
    response = c.response_class(c.sock, method='GET')
    try:
        response.begin()
        actual_status = response.status
        assert actual_status == HTTP_BAD_REQUEST
        actual_resp_body = response.read(22)
        assert actual_resp_body == b'Malformed Request-Line'
        c.close()
    except socket.error as ex:
        if ex.errno != errno.ECONNRESET:
            raise