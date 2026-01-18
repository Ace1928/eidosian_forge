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
def single_exception_from_error_tuple(exc_type, exc_value, tb, client_options=None, mechanism=None, exception_id=None, parent_id=None, source=None):
    """
    Creates a dict that goes into the events `exception.values` list and is ingestible by Sentry.

    See the Exception Interface documentation for more details:
    https://develop.sentry.dev/sdk/event-payloads/exception/
    """
    exception_value = {}
    exception_value['mechanism'] = mechanism.copy() if mechanism else {'type': 'generic', 'handled': True}
    if exception_id is not None:
        exception_value['mechanism']['exception_id'] = exception_id
    if exc_value is not None:
        errno = get_errno(exc_value)
    else:
        errno = None
    if errno is not None:
        exception_value['mechanism'].setdefault('meta', {}).setdefault('errno', {}).setdefault('number', errno)
    if source is not None:
        exception_value['mechanism']['source'] = source
    is_root_exception = exception_id == 0
    if not is_root_exception and parent_id is not None:
        exception_value['mechanism']['parent_id'] = parent_id
        exception_value['mechanism']['type'] = 'chained'
    if is_root_exception and 'type' not in exception_value['mechanism']:
        exception_value['mechanism']['type'] = 'generic'
    is_exception_group = BaseExceptionGroup is not None and isinstance(exc_value, BaseExceptionGroup)
    if is_exception_group:
        exception_value['mechanism']['is_exception_group'] = True
    exception_value['module'] = get_type_module(exc_type)
    exception_value['type'] = get_type_name(exc_type)
    exception_value['value'] = get_error_message(exc_value)
    if client_options is None:
        include_local_variables = True
        include_source_context = True
        max_value_length = DEFAULT_MAX_VALUE_LENGTH
    else:
        include_local_variables = client_options['include_local_variables']
        include_source_context = client_options['include_source_context']
        max_value_length = client_options['max_value_length']
    frames = [serialize_frame(tb.tb_frame, tb_lineno=tb.tb_lineno, include_local_variables=include_local_variables, include_source_context=include_source_context, max_value_length=max_value_length) for tb in iter_stacks(tb)]
    if frames:
        exception_value['stacktrace'] = {'frames': frames}
    return exception_value