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
def test_manual_redirect_with_partial_body_read(self, httpbin):
    s = requests.Session()
    r1 = s.get(httpbin('redirect/2'), allow_redirects=False, stream=True)
    assert r1.is_redirect
    rg = s.resolve_redirects(r1, r1.request, stream=True)
    r1.iter_content(8)
    r2 = next(rg)
    assert r2.is_redirect
    for _ in r2.iter_content():
        pass
    r3 = next(rg)
    assert not r3.is_redirect