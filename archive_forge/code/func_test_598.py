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
@pytest.mark.xfail(reason='Sometimes this test fails due to low timeout. Ref: https://github.com/cherrypy/cherrypy/issues/598')
def test_598(test_client):
    """Test serving large file with a read timeout in place."""
    conn = test_client.get_connection()
    remote_data_conn = urllib.request.urlopen('%s://%s:%s/one_megabyte_of_a' % ('http', conn.host, conn.port))
    buf = remote_data_conn.read(512)
    time.sleep(timeout * 0.6)
    remaining = 1024 * 1024 - 512
    while remaining:
        data = remote_data_conn.read(remaining)
        if not data:
            break
        buf += data
        remaining -= len(data)
    assert len(buf) == 1024 * 1024
    assert buf == b'a' * 1024 * 1024
    assert remaining == 0
    remote_data_conn.close()