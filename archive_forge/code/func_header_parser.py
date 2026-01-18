from __future__ import annotations
import sys
from http.client import responses
from typing import TYPE_CHECKING
from vine import Thenable, maybe_promise, promise
from kombu.exceptions import HttpError
from kombu.utils.compat import coro
from kombu.utils.encoding import bytes_to_str
from kombu.utils.functional import maybe_list, memoize
@coro
def header_parser(keyt=normalize_header):
    while 1:
        line, headers = (yield)
        if line.startswith('HTTP/'):
            continue
        elif not line:
            headers.complete = True
            continue
        elif line[0].isspace():
            pkey = headers._prev_key
            headers[pkey] = ' '.join([headers.get(pkey) or '', line.lstrip()])
        else:
            key, value = line.split(':', 1)
            key = headers._prev_key = keyt(key)
            headers[key] = value.strip()