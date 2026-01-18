import os
import binascii
from typing import List
from libcloud.utils.retry import DEFAULT_DELAY  # noqa: F401
from libcloud.utils.retry import DEFAULT_BACKOFF  # noqa: F401
from libcloud.utils.retry import DEFAULT_TIMEOUT  # noqa: F401
from libcloud.utils.retry import TRANSIENT_SSL_ERROR  # noqa: F401
from libcloud.utils.retry import Retry  # flake8: noqa
from libcloud.utils.retry import TransientSSLError  # noqa: F401
from libcloud.common.providers import get_driver as _get_driver
from libcloud.common.providers import set_driver as _set_driver
def merge_valid_keys(params, valid_keys, extra):
    """
    Merge valid keys from extra into params dictionary and return
    dictionary with keys which have been merged.

    Note: params is modified in place.
    """
    merged = {}
    if not extra:
        return merged
    for key in valid_keys:
        if key in extra:
            params[key] = extra[key]
            merged[key] = extra[key]
    return merged