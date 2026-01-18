import contextlib
import re
import socket
import ssl
import zlib
from base64 import b64decode
from io import BufferedReader, BytesIO, TextIOWrapper
from test import onlyBrotlipy
import mock
import pytest
import six
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.response import HTTPResponse, brotli
from urllib3.util.response import is_fp_closed
from urllib3.util.retry import RequestHistory, Retry
def test_non_timeout_ssl_error_on_read(self):
    mac_error = ssl.SSLError('SSL routines', 'ssl3_get_record', 'decryption failed or bad record mac')

    @contextlib.contextmanager
    def make_bad_mac_fp():
        fp = BytesIO(b'')
        with mock.patch.object(fp, 'read') as fp_read:
            fp_read.side_effect = mac_error
            yield fp
    with make_bad_mac_fp() as fp:
        with pytest.raises(SSLError) as e:
            HTTPResponse(fp)
        assert e.value.args[0] == mac_error
    with make_bad_mac_fp() as fp:
        resp = HTTPResponse(fp, preload_content=False)
        with pytest.raises(SSLError) as e:
            resp.read()
        assert e.value.args[0] == mac_error