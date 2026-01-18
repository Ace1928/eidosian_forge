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
def get_default_release():
    """Try to guess a default release."""
    release = os.environ.get('SENTRY_RELEASE')
    if release:
        return release
    release = get_git_revision()
    if release:
        return release
    for var in ('HEROKU_SLUG_COMMIT', 'SOURCE_VERSION', 'CODEBUILD_RESOLVED_SOURCE_VERSION', 'CIRCLE_SHA1', 'GAE_DEPLOYMENT_ID'):
        release = os.environ.get(var)
        if release:
            return release
    return None