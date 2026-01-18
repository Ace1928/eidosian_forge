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
def decode_token(token: str, signed: bool=True) -> dict[str, Any]:
    """
    Decodes a signed or unsigned JWT token.
    """
    if signed and '.' in token:
        signing_input, _ = token.encode('utf-8').rsplit(b'.', 1)
        _, payload_segment = signing_input.split(b'.', 1)
    else:
        payload_segment = token
    return json.loads(base64url_decode(payload_segment).decode('utf-8'))