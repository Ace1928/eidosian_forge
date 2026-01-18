import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def strip_version(endpoint):
    """Strip version from the last component of endpoint if present."""
    if not isinstance(endpoint, str):
        raise ValueError('Expected endpoint')
    version = None
    endpoint = endpoint.rstrip('/')
    url_parts = urllib.parse.urlparse(endpoint)
    scheme, netloc, path, __, __, __ = url_parts
    path = path.lstrip('/')
    if re.match('v\\d+\\.?\\d*', path):
        version = float(path.lstrip('v'))
        endpoint = scheme + '://' + netloc
    return (endpoint, version)