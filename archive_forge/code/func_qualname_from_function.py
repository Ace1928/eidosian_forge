import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
def qualname_from_function(func):
    """Return the qualified name of func. Works with regular function, lambda, partial and partialmethod."""
    func_qualname = None
    try:
        return '%s.%s.%s' % (func.im_class.__module__, func.im_class.__name__, func.__name__)
    except Exception:
        pass
    prefix, suffix = ('', '')
    if _PARTIALMETHOD_AVAILABLE and hasattr(func, '_partialmethod') and isinstance(func._partialmethod, partialmethod):
        prefix, suffix = ('partialmethod(<function ', '>)')
        func = func._partialmethod.func
    elif isinstance(func, partial) and hasattr(func.func, '__name__'):
        prefix, suffix = ('partial(<function ', '>)')
        func = func.func
    if hasattr(func, '__qualname__'):
        func_qualname = func.__qualname__
    elif hasattr(func, '__name__'):
        func_qualname = func.__name__
    if func_qualname is not None:
        if hasattr(func, '__module__'):
            func_qualname = func.__module__ + '.' + func_qualname
        func_qualname = prefix + func_qualname + suffix
    return func_qualname