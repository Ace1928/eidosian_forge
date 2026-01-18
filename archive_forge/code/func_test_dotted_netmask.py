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
@pytest.mark.parametrize('mask, expected', ((8, '255.0.0.0'), (24, '255.255.255.0'), (25, '255.255.255.128')))
def test_dotted_netmask(mask, expected):
    assert dotted_netmask(mask) == expected