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
def test_auth_is_stripped_on_http_downgrade(self, httpbin, httpbin_secure, httpbin_ca_bundle):
    r = requests.get(httpbin_secure('redirect-to'), params={'url': httpbin('get')}, auth=('user', 'pass'), verify=httpbin_ca_bundle)
    assert r.history[0].request.headers['Authorization']
    assert 'Authorization' not in r.request.headers