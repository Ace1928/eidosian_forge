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
def test_POSTBIN_GET_POST_FILES_WITH_DATA(self, httpbin):
    url = httpbin('post')
    requests.post(url).raise_for_status()
    post1 = requests.post(url, data={'some': 'data'})
    assert post1.status_code == 200
    with open('requirements-dev.txt') as f:
        post2 = requests.post(url, data={'some': 'data'}, files={'some': f})
    assert post2.status_code == 200
    post4 = requests.post(url, data='[{"some": "json"}]')
    assert post4.status_code == 200
    with pytest.raises(ValueError):
        requests.post(url, files=['bad file data'])