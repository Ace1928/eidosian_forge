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
@pytest.mark.parametrize('exception, args, expected', ((urllib3.exceptions.ProtocolError, tuple(), ChunkedEncodingError), (urllib3.exceptions.DecodeError, tuple(), ContentDecodingError), (urllib3.exceptions.ReadTimeoutError, (None, '', ''), ConnectionError), (urllib3.exceptions.SSLError, tuple(), RequestsSSLError)))
def test_iter_content_wraps_exceptions(self, httpbin, mocker, exception, args, expected):
    r = requests.Response()
    r.raw = mocker.Mock()
    r.raw.stream.side_effect = exception(*args)
    with pytest.raises(expected):
        next(r.iter_content(1024))