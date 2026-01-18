import os
import copy
import filecmp
from io import BytesIO
import tarfile
import zipfile
from collections import deque
import pytest
from requests import compat
from requests.cookies import RequestsCookieJar
from requests.structures import CaseInsensitiveDict
from requests.utils import (
from requests._internal_utils import unicode_is_ascii
from .compat import StringIO, cStringIO
@pytest.mark.parametrize('url, auth', (('http://' + ENCODED_USER + ':' + ENCODED_PASSWORD + '@' + 'request.com/url.html#test', (USER, PASSWORD)), ('http://user:pass@complex.url.com/path?query=yes', ('user', 'pass')), ('http://user:pass%20pass@complex.url.com/path?query=yes', ('user', 'pass pass')), ('http://user:pass pass@complex.url.com/path?query=yes', ('user', 'pass pass')), ('http://user%25user:pass@complex.url.com/path?query=yes', ('user%user', 'pass')), ('http://user:pass%23pass@complex.url.com/path?query=yes', ('user', 'pass#pass')), ('http://complex.url.com/path?query=yes', ('', ''))))
def test_get_auth_from_url(url, auth):
    assert get_auth_from_url(url) == auth