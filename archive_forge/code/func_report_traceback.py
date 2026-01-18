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
def report_traceback():
    """
    Reports a timestamp and full traceback for a given exception.

    :return: Full traceback and timestamp.
    """
    try:
        formatted_lines = traceback.format_exc()
        now = time.time()
        return (formatted_lines, now)
    except AttributeError:
        return (None, None)