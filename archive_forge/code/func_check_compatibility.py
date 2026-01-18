import urllib3
import warnings
from .exceptions import RequestsDependencyWarning
from urllib3.exceptions import DependencyWarning
from .__version__ import __title__, __description__, __url__, __version__
from .__version__ import __build__, __author__, __author_email__, __license__
from .__version__ import __copyright__, __cake__
from . import utils
from . import packages
from .models import Request, Response, PreparedRequest
from .api import request, get, head, post, patch, put, delete, options
from .sessions import session, Session
from .status_codes import codes
from .exceptions import (
import logging
from logging import NullHandler
def check_compatibility(urllib3_version, chardet_version, charset_normalizer_version):
    urllib3_version = urllib3_version.split('.')
    assert urllib3_version != ['dev']
    if len(urllib3_version) == 2:
        urllib3_version.append('0')
    major, minor, patch = urllib3_version
    major, minor, patch = (int(major), int(minor), int(patch))
    assert major == 1
    assert minor >= 21
    assert minor <= 26
    if chardet_version:
        major, minor, patch = chardet_version.split('.')[:3]
        major, minor, patch = (int(major), int(minor), int(patch))
        assert (3, 0, 2) <= (major, minor, patch) < (5, 0, 0)
    elif charset_normalizer_version:
        major, minor, patch = charset_normalizer_version.split('.')[:3]
        major, minor, patch = (int(major), int(minor), int(patch))
        assert (2, 0, 0) <= (major, minor, patch) < (3, 0, 0)
    else:
        raise Exception('You need either charset_normalizer or chardet installed')