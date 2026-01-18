import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
def no_network():
    """Return true if the NO_NETWORK environment variable is set.
    Can be used to skip relevant tests.
    """
    return 'NO_NETWORK' in os.environ