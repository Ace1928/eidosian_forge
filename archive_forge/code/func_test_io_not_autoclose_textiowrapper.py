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
def test_io_not_autoclose_textiowrapper(self):
    fp = BytesIO(b'\xc3\xa4\xc3\xb6\xc3\xbc\xc3\x9f\n\xce\xb1\xce\xb2\xce\xb3\xce\xb4')
    resp = HTTPResponse(fp, preload_content=False, auto_close=False)
    if six.PY2:
        resp = BufferedReader(resp)
    reader = TextIOWrapper(resp, encoding='utf8')
    assert list(reader) == [u'äöüß\n', u'αβγδ']
    assert not reader.closed
    assert not resp.closed
    with pytest.raises(StopIteration):
        next(reader)
    reader.close()
    assert reader.closed
    assert resp.closed
    with pytest.raises(ValueError) as ctx:
        next(reader)
    assert re.match('I/O operation on closed file.?', str(ctx.value))