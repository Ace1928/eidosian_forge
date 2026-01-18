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
@pytest.mark.parametrize('scheme', ('http://', 'HTTP://', 'hTTp://', 'HttP://'))
def test_mixed_case_scheme_acceptable(self, httpbin, scheme):
    s = requests.Session()
    s.proxies = getproxies()
    parts = urlparse(httpbin('get'))
    url = scheme + parts.netloc + parts.path
    r = requests.Request('GET', url)
    r = s.send(r.prepare())
    assert r.status_code == 200, 'failed for scheme {}'.format(scheme)