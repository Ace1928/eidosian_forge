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
def test_buggy_incomplete_read(self):
    content_length = 1337
    fp = BytesIO(b'')
    resp = HTTPResponse(fp, headers={'content-length': str(content_length)}, preload_content=False, enforce_content_length=True)
    with pytest.raises(ProtocolError) as ctx:
        resp.read(3)
    orig_ex = ctx.value.args[1]
    assert isinstance(orig_ex, IncompleteRead)
    assert orig_ex.partial == 0
    assert orig_ex.expected == content_length