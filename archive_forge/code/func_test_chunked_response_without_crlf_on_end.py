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
def test_chunked_response_without_crlf_on_end(self):
    stream = [b'foo', b'bar', b'baz']
    fp = MockChunkedEncodingWithoutCRLFOnEnd(stream)
    r = httplib.HTTPResponse(MockSock)
    r.fp = fp
    r.chunked = True
    r.chunk_left = None
    resp = HTTPResponse(r, preload_content=False, headers={'transfer-encoding': 'chunked'})
    assert stream == list(resp.stream())