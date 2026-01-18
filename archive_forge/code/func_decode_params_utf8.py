from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
def decode_params_utf8(params):
    """Ensures that all parameters in a list of 2-element tuples are decoded to

    unicode using UTF-8.
    """
    decoded = []
    for k, v in params:
        decoded.append((k.decode('utf-8') if isinstance(k, bytes) else k, v.decode('utf-8') if isinstance(v, bytes) else v))
    return decoded