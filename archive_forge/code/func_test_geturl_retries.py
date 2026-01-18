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
def test_geturl_retries(self):
    fp = BytesIO(b'')
    resp = HTTPResponse(fp, request_url='http://example.com')
    request_histories = [RequestHistory(method='GET', url='http://example.com', error=None, status=301, redirect_location='https://example.com/'), RequestHistory(method='GET', url='https://example.com/', error=None, status=301, redirect_location='https://www.example.com')]
    retry = Retry(history=request_histories)
    resp = HTTPResponse(fp, retries=retry)
    assert resp.geturl() == 'https://www.example.com'