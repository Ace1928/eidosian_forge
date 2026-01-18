from __future__ import division
import json
import os
import pickle
import collections
import contextlib
import warnings
import re
import io
import requests
import pytest
import urllib3
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
from requests.cookies import (
from requests.exceptions import (
from requests.exceptions import SSLError as RequestsSSLError
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict
from requests.sessions import SessionRedirectMixin
from requests.models import urlencode
from requests.hooks import default_hooks
from requests.compat import JSONDecodeError, is_py3, MutableMapping
from .compat import StringIO, u
from .utils import override_environ
from urllib3.util import Timeout as Urllib3Timeout
def test_headers_preserve_order(self, httpbin):
    """Preserve order when headers provided as OrderedDict."""
    ses = requests.Session()
    ses.headers = collections.OrderedDict()
    ses.headers['Accept-Encoding'] = 'identity'
    ses.headers['First'] = '1'
    ses.headers['Second'] = '2'
    headers = collections.OrderedDict([('Third', '3'), ('Fourth', '4')])
    headers['Fifth'] = '5'
    headers['Second'] = '222'
    req = requests.Request('GET', httpbin('get'), headers=headers)
    prep = ses.prepare_request(req)
    items = list(prep.headers.items())
    assert items[0] == ('Accept-Encoding', 'identity')
    assert items[1] == ('First', '1')
    assert items[2] == ('Second', '222')
    assert items[3] == ('Third', '3')
    assert items[4] == ('Fourth', '4')
    assert items[5] == ('Fifth', '5')