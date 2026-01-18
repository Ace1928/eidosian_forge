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
def is_valid_sample_rate(rate, source):
    """
    Checks the given sample rate to make sure it is valid type and value (a
    boolean or a number between 0 and 1, inclusive).
    """
    if not isinstance(rate, (Real, Decimal)) or math.isnan(rate):
        logger.warning('{source} Given sample rate is invalid. Sample rate must be a boolean or a number between 0 and 1. Got {rate} of type {type}.'.format(source=source, rate=rate, type=type(rate)))
        return False
    rate = float(rate)
    if rate < 0 or rate > 1:
        logger.warning('{source} Given sample rate is invalid. Sample rate must be between 0 and 1. Got {rate}.'.format(source=source, rate=rate))
        return False
    return True