from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def call_with_backoff(fn, **kwargs):
    if 'retry_strategy' not in kwargs:
        kwargs['retry_strategy'] = _get_retry_strategy()
    try:
        return fn(**kwargs)
    except TypeError as te:
        if 'unexpected keyword argument' in str(te):
            del kwargs['retry_strategy']
            return fn(**kwargs)
        else:
            raise