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
def parse_version(version):
    """
    Parses a version string into a tuple of integers.
    This uses the parsing loging from PEP 440:
    https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
    """
    VERSION_PATTERN = '  # noqa: N806\n        v?\n        (?:\n            (?:(?P<epoch>[0-9]+)!)?                           # epoch\n            (?P<release>[0-9]+(?:\\.[0-9]+)*)                  # release segment\n            (?P<pre>                                          # pre-release\n                [-_\\.]?\n                (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))\n                [-_\\.]?\n                (?P<pre_n>[0-9]+)?\n            )?\n            (?P<post>                                         # post release\n                (?:-(?P<post_n1>[0-9]+))\n                |\n                (?:\n                    [-_\\.]?\n                    (?P<post_l>post|rev|r)\n                    [-_\\.]?\n                    (?P<post_n2>[0-9]+)?\n                )\n            )?\n            (?P<dev>                                          # dev release\n                [-_\\.]?\n                (?P<dev_l>dev)\n                [-_\\.]?\n                (?P<dev_n>[0-9]+)?\n            )?\n        )\n        (?:\\+(?P<local>[a-z0-9]+(?:[-_\\.][a-z0-9]+)*))?       # local version\n    '
    pattern = re.compile('^\\s*' + VERSION_PATTERN + '\\s*$', re.VERBOSE | re.IGNORECASE)
    try:
        release = pattern.match(version).groupdict()['release']
        release_tuple = tuple(map(int, release.split('.')[:3]))
    except (TypeError, ValueError, AttributeError):
        return None
    return release_tuple