from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def apply_diff_key(src, dest, klist):
    diff_cnt = 0
    for k in klist:
        v = src.get(k)
        if v is not None and v != dest.get(k):
            dest[k] = v
            diff_cnt = diff_cnt + 1
    return diff_cnt