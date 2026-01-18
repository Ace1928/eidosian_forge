import base64
from calendar import timegm
from collections.abc import Mapping
import gzip
import hashlib
import hmac
import io
import json
import logging
import time
import traceback
def split_request_headers(options, prefix=''):
    headers = {}
    if isinstance(options, Mapping):
        options = options.items()
    for item in options:
        if isinstance(item, str):
            if ':' not in item:
                raise ValueError("Metadata parameter %s must contain a ':'.\nExample: 'Color:Blue' or 'Size:Large'" % item)
            item = item.split(':', 1)
        if len(item) != 2:
            raise ValueError("Metadata parameter %r must have exactly two items.\nExample: ('Color', 'Blue') or ['Size', 'Large']" % (item,))
        headers[(prefix + item[0]).title()] = item[1].strip()
    return headers