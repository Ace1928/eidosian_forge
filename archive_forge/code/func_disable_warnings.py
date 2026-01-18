from __future__ import absolute_import
import logging
import warnings
from logging import NullHandler
from . import exceptions
from ._version import __version__
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, connection_from_url
from .filepost import encode_multipart_formdata
from .poolmanager import PoolManager, ProxyManager, proxy_from_url
from .response import HTTPResponse
from .util.request import make_headers
from .util.retry import Retry
from .util.timeout import Timeout
from .util.url import get_host
def disable_warnings(category=exceptions.HTTPWarning):
    """
    Helper for quickly disabling all urllib3 warnings.
    """
    warnings.simplefilter('ignore', category)