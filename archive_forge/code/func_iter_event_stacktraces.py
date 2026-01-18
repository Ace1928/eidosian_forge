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
def iter_event_stacktraces(event):
    if 'stacktrace' in event:
        yield event['stacktrace']
    if 'threads' in event:
        for thread in event['threads'].get('values') or ():
            if 'stacktrace' in thread:
                yield thread['stacktrace']
    if 'exception' in event:
        for exception in event['exception'].get('values') or ():
            if 'stacktrace' in exception:
                yield exception['stacktrace']