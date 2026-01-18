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
def str2list(data):
    """
    Create a list of values from a whitespace and newline delimited text
    (keys are ignored).

    For example, this:
    ip 1.2.3.4
    ip 1.2.3.5
    ip 1.2.3.6

    becomes:
    ['1.2.3.4', '1.2.3.5', '1.2.3.6']
    """
    list_data = []
    for line in data.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            split = line.split(' ')
            value = split[1]
        except Exception:
            continue
        list_data.append(value)
    return list_data