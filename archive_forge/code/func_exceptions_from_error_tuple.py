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
def exceptions_from_error_tuple(exc_info, client_options=None, mechanism=None):
    exc_type, exc_value, tb = exc_info
    is_exception_group = BaseExceptionGroup is not None and isinstance(exc_value, BaseExceptionGroup)
    if is_exception_group:
        _, exceptions = exceptions_from_error(exc_type=exc_type, exc_value=exc_value, tb=tb, client_options=client_options, mechanism=mechanism, exception_id=0, parent_id=0)
    else:
        exceptions = []
        for exc_type, exc_value, tb in walk_exception_chain(exc_info):
            exceptions.append(single_exception_from_error_tuple(exc_type, exc_value, tb, client_options, mechanism))
    exceptions.reverse()
    return exceptions