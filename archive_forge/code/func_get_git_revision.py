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
def get_git_revision():
    try:
        with open(os.path.devnull, 'w+') as null:
            revision = subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, stderr=null, stdin=null).communicate()[0].strip().decode('utf-8')
    except (OSError, IOError, FileNotFoundError):
        return None
    return revision