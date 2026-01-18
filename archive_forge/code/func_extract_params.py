from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
def extract_params(raw):
    """Extract parameters and return them as a list of 2-tuples.

    Will successfully extract parameters from urlencoded query strings,
    dicts, or lists of 2-tuples. Empty strings/dicts/lists will return an
    empty list of parameters. Any other input will result in a return
    value of None.
    """
    if isinstance(raw, bytes) or isinstance(raw, unicode_type):
        try:
            params = urldecode(raw)
        except ValueError:
            params = None
    elif hasattr(raw, '__iter__'):
        try:
            dict(raw)
        except ValueError:
            params = None
        except TypeError:
            params = None
        else:
            params = list(raw.items() if isinstance(raw, dict) else raw)
            params = decode_params_utf8(params)
    else:
        params = None
    return params