import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_content_length_required(test_client):
    """Test POST query with body failing because of missing Content-Length."""
    c = test_client.get_connection()
    c.request('POST', '/body_required')
    response = c.getresponse()
    response.read()
    actual_status = response.status
    assert actual_status == HTTP_LENGTH_REQUIRED
    c.close()