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
def test_header_value_not_str(self, httpbin):
    """Ensure the header value is of type string or bytes as
        per discussion in GH issue #3386
        """
    headers_int = {'foo': 3}
    headers_dict = {'bar': {'foo': 'bar'}}
    headers_list = {'baz': ['foo', 'bar']}
    with pytest.raises(InvalidHeader) as excinfo:
        r = requests.get(httpbin('get'), headers=headers_int)
    assert 'foo' in str(excinfo.value)
    with pytest.raises(InvalidHeader) as excinfo:
        r = requests.get(httpbin('get'), headers=headers_dict)
    assert 'bar' in str(excinfo.value)
    with pytest.raises(InvalidHeader) as excinfo:
        r = requests.get(httpbin('get'), headers=headers_list)
    assert 'baz' in str(excinfo.value)