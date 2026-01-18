import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
def instance_cache(func):
    """Method decorator for caching method calls to a single instance.

    **This is not a general purpose caching decorator.**

    In order to use this, you *must* provide an ``_instance_cache``
    attribute on the instance.

    This decorator is used to cache method calls.  The cache is only
    scoped to a single instance though such that multiple instances
    will maintain their own cache.  In order to keep things simple,
    this decorator requires that you provide an ``_instance_cache``
    attribute on your instance.

    """
    func_name = func.__name__

    @functools.wraps(func)
    def _cache_guard(self, *args, **kwargs):
        cache_key = (func_name, args)
        if kwargs:
            kwarg_items = tuple(sorted(kwargs.items()))
            cache_key = (func_name, args, kwarg_items)
        result = self._instance_cache.get(cache_key)
        if result is not None:
            return result
        result = func(self, *args, **kwargs)
        self._instance_cache[cache_key] = result
        return result
    return _cache_guard