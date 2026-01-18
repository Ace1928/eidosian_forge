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
def test_custom_redirect_mixin(self, httpbin):
    """Tests a custom mixin to overwrite ``get_redirect_target``.

        Ensures a subclassed ``requests.Session`` can handle a certain type of
        malformed redirect responses.

        1. original request receives a proper response: 302 redirect
        2. following the redirect, a malformed response is given:
            status code = HTTP 200
            location = alternate url
        3. the custom session catches the edge case and follows the redirect
        """
    url_final = httpbin('html')
    querystring_malformed = urlencode({'location': url_final})
    url_redirect_malformed = httpbin('response-headers?%s' % querystring_malformed)
    querystring_redirect = urlencode({'url': url_redirect_malformed})
    url_redirect = httpbin('redirect-to?%s' % querystring_redirect)
    urls_test = [url_redirect, url_redirect_malformed, url_final]

    class CustomRedirectSession(requests.Session):

        def get_redirect_target(self, resp):
            if resp.is_redirect:
                return resp.headers['location']
            location = resp.headers.get('location')
            if location and location != resp.url:
                return location
            return None
    session = CustomRedirectSession()
    r = session.get(urls_test[0])
    assert len(r.history) == 2
    assert r.status_code == 200
    assert r.history[0].status_code == 302
    assert r.history[0].is_redirect
    assert r.history[1].status_code == 200
    assert not r.history[1].is_redirect
    assert r.url == urls_test[2]