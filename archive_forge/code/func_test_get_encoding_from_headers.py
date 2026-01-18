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
@pytest.mark.parametrize('value, expected', ((CaseInsensitiveDict(), None), (CaseInsensitiveDict({'content-type': 'application/json; charset=utf-8'}), 'utf-8'), (CaseInsensitiveDict({'content-type': 'text/plain'}), 'ISO-8859-1')))
def test_get_encoding_from_headers(value, expected):
    assert get_encoding_from_headers(value) == expected