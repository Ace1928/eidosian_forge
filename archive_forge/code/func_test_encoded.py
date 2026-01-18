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
@pytest.mark.parametrize('encoding', ('utf-32', 'utf-8-sig', 'utf-16', 'utf-8', 'utf-16-be', 'utf-16-le', 'utf-32-be', 'utf-32-le'))
def test_encoded(self, encoding):
    data = '{}'.encode(encoding)
    assert guess_json_utf(data) == encoding