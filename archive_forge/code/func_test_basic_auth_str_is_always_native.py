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
@pytest.mark.parametrize('username, password, auth_str', (('test', 'test', 'Basic dGVzdDp0ZXN0'), (u'имя'.encode('utf-8'), u'пароль'.encode('utf-8'), 'Basic 0LjQvNGPOtC/0LDRgNC+0LvRjA==')))
def test_basic_auth_str_is_always_native(self, username, password, auth_str):
    s = _basic_auth_str(username, password)
    assert isinstance(s, builtin_str)
    assert s == auth_str