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
def match_regex_list(item, regex_list=None, substring_matching=False):
    if regex_list is None:
        return False
    for item_matcher in regex_list:
        if not substring_matching and item_matcher[-1] != '$':
            item_matcher += '$'
        matched = re.search(item_matcher, item)
        if matched:
            return True
    return False