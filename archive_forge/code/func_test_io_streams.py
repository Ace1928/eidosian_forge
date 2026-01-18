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
@pytest.mark.parametrize('stream, value', ((StringIO.StringIO, 'Test'), (BytesIO, b'Test'), pytest.param(cStringIO, 'Test', marks=pytest.mark.skipif('cStringIO is None'))))
def test_io_streams(self, stream, value):
    """Ensures that we properly deal with different kinds of IO streams."""
    assert super_len(stream()) == 0
    assert super_len(stream(value)) == 4