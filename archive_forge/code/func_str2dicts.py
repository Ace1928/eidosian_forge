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
def str2dicts(data):
    """
    Create a list of dictionaries from a whitespace and newline delimited text.

    For example, this:
    cpu 1100
    ram 640

    cpu 2200
    ram 1024

    becomes:
    [{'cpu': '1100', 'ram': '640'}, {'cpu': '2200', 'ram': '1024'}]
    """
    list_data = []
    list_data.append({})
    d = list_data[-1]
    lines = data.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            d = {}
            list_data.append(d)
            d = list_data[-1]
            continue
        whitespace = line.find(' ')
        if not whitespace:
            continue
        key = line[0:whitespace]
        value = line[whitespace + 1:]
        d.update({key: value})
    list_data = [val for val in list_data if val != {}]
    return list_data