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
def parse_query(query: str) -> dict[str, Any]:
    """
    Parses a url query string, e.g. ?a=1&b=2.1&c=string, converting
    numeric strings to int or float types.
    """
    query_dict = dict(urlparse.parse_qsl(query[1:]))
    parsed_query: dict[str, Any] = {}
    for k, v in query_dict.items():
        if v.isdigit():
            parsed_query[k] = int(v)
        elif is_number(v):
            parsed_query[k] = float(v)
        elif v.startswith(('[', '{')):
            try:
                parsed_query[k] = json.loads(v)
            except Exception:
                try:
                    parsed_query[k] = ast.literal_eval(v)
                except Exception:
                    log.warning(f'Could not parse value {v!r} of query parameter {k}. Parameter will be ignored.')
        elif v.lower() in ('true', 'false'):
            parsed_query[k] = v.lower() == 'true'
        else:
            parsed_query[k] = v
    return parsed_query