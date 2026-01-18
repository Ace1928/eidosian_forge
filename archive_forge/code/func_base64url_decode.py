from __future__ import annotations
import ast
import base64
import datetime as dt
import json
import logging
import numbers
import os
import pathlib
import re
import sys
import urllib.parse as urlparse
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping, MutableSequence
from datetime import datetime
from functools import partial
from html import escape  # noqa
from importlib import import_module
from typing import Any, AnyStr
import bokeh
import numpy as np
import param
from bokeh.core.has_props import _default_resolver
from bokeh.model import Model
from packaging.version import Version
from .checks import (  # noqa
from .parameters import (  # noqa
def base64url_decode(input):
    if isinstance(input, str):
        input = input.encode('ascii')
    rem = len(input) % 4
    if rem > 0:
        input += b'=' * (4 - rem)
    return base64.urlsafe_b64decode(input)