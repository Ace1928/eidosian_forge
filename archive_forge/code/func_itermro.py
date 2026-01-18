import datetime
import numbers
import sys
from base64 import b64decode as base64decode
from base64 import b64encode as base64encode
from functools import partial
from inspect import getmro
from itertools import takewhile
from kombu.utils.encoding import bytes_to_str, safe_repr, str_to_bytes
def itermro(cls, stop):
    return takewhile(lambda sup: sup not in stop, getmro(cls))