from __future__ import annotations
import sys
from http.client import responses
from typing import TYPE_CHECKING
from vine import Thenable, maybe_promise, promise
from kombu.exceptions import HttpError
from kombu.utils.compat import coro
from kombu.utils.encoding import bytes_to_str
from kombu.utils.functional import maybe_list, memoize
@memoize(maxsize=1000)
def normalize_header(key):
    return '-'.join((p.capitalize() for p in key.split('-')))