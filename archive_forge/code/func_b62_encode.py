import base64
import datetime
import json
import time
import warnings
import zlib
from django.conf import settings
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.encoding import force_bytes
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
def b62_encode(s):
    if s == 0:
        return '0'
    sign = '-' if s < 0 else ''
    s = abs(s)
    encoded = ''
    while s > 0:
        s, remainder = divmod(s, 62)
        encoded = BASE62_ALPHABET[remainder] + encoded
    return sign + encoded