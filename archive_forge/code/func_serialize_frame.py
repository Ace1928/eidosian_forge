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
def serialize_frame(frame, tb_lineno=None, include_local_variables=True, include_source_context=True, max_value_length=None):
    f_code = getattr(frame, 'f_code', None)
    if not f_code:
        abs_path = None
        function = None
    else:
        abs_path = frame.f_code.co_filename
        function = frame.f_code.co_name
    try:
        module = frame.f_globals['__name__']
    except Exception:
        module = None
    if tb_lineno is None:
        tb_lineno = frame.f_lineno
    rv = {'filename': filename_for_module(module, abs_path) or None, 'abs_path': os.path.abspath(abs_path) if abs_path else None, 'function': function or '<unknown>', 'module': module, 'lineno': tb_lineno}
    if include_source_context:
        rv['pre_context'], rv['context_line'], rv['post_context'] = get_source_context(frame, tb_lineno, max_value_length)
    if include_local_variables:
        rv['vars'] = copy(frame.f_locals)
    return rv