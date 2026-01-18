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
def get_source_context(frame, tb_lineno, max_value_length=None):
    try:
        abs_path = frame.f_code.co_filename
    except Exception:
        abs_path = None
    try:
        module = frame.f_globals['__name__']
    except Exception:
        return ([], None, [])
    try:
        loader = frame.f_globals['__loader__']
    except Exception:
        loader = None
    lineno = tb_lineno - 1
    if lineno is not None and abs_path:
        return get_lines_from_file(abs_path, lineno, max_value_length, loader=loader, module=module)
    return ([], None, [])