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
def exceptions_from_error(exc_type, exc_value, tb, client_options=None, mechanism=None, exception_id=0, parent_id=0, source=None):
    """
    Creates the list of exceptions.
    This can include chained exceptions and exceptions from an ExceptionGroup.

    See the Exception Interface documentation for more details:
    https://develop.sentry.dev/sdk/event-payloads/exception/
    """
    parent = single_exception_from_error_tuple(exc_type=exc_type, exc_value=exc_value, tb=tb, client_options=client_options, mechanism=mechanism, exception_id=exception_id, parent_id=parent_id, source=source)
    exceptions = [parent]
    parent_id = exception_id
    exception_id += 1
    should_supress_context = hasattr(exc_value, '__suppress_context__') and exc_value.__suppress_context__
    if should_supress_context:
        exception_has_cause = exc_value and hasattr(exc_value, '__cause__') and (exc_value.__cause__ is not None)
        if exception_has_cause:
            cause = exc_value.__cause__
            exception_id, child_exceptions = exceptions_from_error(exc_type=type(cause), exc_value=cause, tb=getattr(cause, '__traceback__', None), client_options=client_options, mechanism=mechanism, exception_id=exception_id, source='__cause__')
            exceptions.extend(child_exceptions)
    else:
        exception_has_content = exc_value and hasattr(exc_value, '__context__') and (exc_value.__context__ is not None)
        if exception_has_content:
            context = exc_value.__context__
            exception_id, child_exceptions = exceptions_from_error(exc_type=type(context), exc_value=context, tb=getattr(context, '__traceback__', None), client_options=client_options, mechanism=mechanism, exception_id=exception_id, source='__context__')
            exceptions.extend(child_exceptions)
    is_exception_group = exc_value and hasattr(exc_value, 'exceptions')
    if is_exception_group:
        for idx, e in enumerate(exc_value.exceptions):
            exception_id, child_exceptions = exceptions_from_error(exc_type=type(e), exc_value=e, tb=getattr(e, '__traceback__', None), client_options=client_options, mechanism=mechanism, exception_id=exception_id, parent_id=parent_id, source='exceptions[%s]' % idx)
            exceptions.extend(child_exceptions)
    return (exception_id, exceptions)