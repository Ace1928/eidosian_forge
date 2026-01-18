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
def test_cookielib_cookiejar_on_redirect(self, httpbin):
    """Tests resolve_redirect doesn't fail when merging cookies
        with non-RequestsCookieJar cookiejar.

        See GH #3579
        """
    cj = cookiejar_from_dict({'foo': 'bar'}, cookielib.CookieJar())
    s = requests.Session()
    s.cookies = cookiejar_from_dict({'cookie': 'tasty'})
    req = requests.Request('GET', httpbin('headers'), cookies=cj)
    prep_req = req.prepare()
    resp = s.send(prep_req)
    resp.status_code = 302
    resp.headers['location'] = httpbin('get')
    redirects = s.resolve_redirects(resp, prep_req)
    resp = next(redirects)
    assert isinstance(prep_req._cookies, cookielib.CookieJar)
    assert isinstance(resp.request._cookies, cookielib.CookieJar)
    assert not isinstance(resp.request._cookies, requests.cookies.RequestsCookieJar)
    cookies = {}
    for c in resp.request._cookies:
        cookies[c.name] = c.value
    assert cookies['foo'] == 'bar'
    assert cookies['cookie'] == 'tasty'