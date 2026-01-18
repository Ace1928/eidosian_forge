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
def test_cookie_parameters(self):
    key = 'some_cookie'
    value = 'some_value'
    secure = True
    domain = 'test.com'
    rest = {'HttpOnly': True}
    jar = requests.cookies.RequestsCookieJar()
    jar.set(key, value, secure=secure, domain=domain, rest=rest)
    assert len(jar) == 1
    assert 'some_cookie' in jar
    cookie = list(jar)[0]
    assert cookie.secure == secure
    assert cookie.domain == domain
    assert cookie._rest['HttpOnly'] == rest['HttpOnly']