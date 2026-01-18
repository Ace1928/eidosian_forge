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
def test_cookie_as_dict_items(self):
    key = 'some_cookie'
    value = 'some_value'
    key1 = 'some_cookie1'
    value1 = 'some_value1'
    jar = requests.cookies.RequestsCookieJar()
    jar.set(key, value)
    jar.set(key1, value1)
    items = jar.items()
    assert items == list(items)
    assert list(items) == list(items)