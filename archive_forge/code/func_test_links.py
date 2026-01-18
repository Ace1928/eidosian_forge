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
def test_links(self):
    r = requests.Response()
    r.headers = {'cache-control': 'public, max-age=60, s-maxage=60', 'connection': 'keep-alive', 'content-encoding': 'gzip', 'content-type': 'application/json; charset=utf-8', 'date': 'Sat, 26 Jan 2013 16:47:56 GMT', 'etag': '"6ff6a73c0e446c1f61614769e3ceb778"', 'last-modified': 'Sat, 26 Jan 2013 16:22:39 GMT', 'link': '<https://api.github.com/users/kennethreitz/repos?page=2&per_page=10>; rel="next", <https://api.github.com/users/kennethreitz/repos?page=7&per_page=10>;  rel="last"', 'server': 'GitHub.com', 'status': '200 OK', 'vary': 'Accept', 'x-content-type-options': 'nosniff', 'x-github-media-type': 'github.beta', 'x-ratelimit-limit': '60', 'x-ratelimit-remaining': '57'}
    assert r.links['next']['rel'] == 'next'