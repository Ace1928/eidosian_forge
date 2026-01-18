from __future__ import annotations
import collections
import datetime as dt
import functools
import inspect
import json
import re
import typing
import warnings
from collections.abc import Mapping
from email.utils import format_datetime, parsedate_to_datetime
from pprint import pprint as py_pprint
from marshmallow.base import FieldABC
from marshmallow.exceptions import FieldInstanceResolutionError
from marshmallow.warnings import RemovedInMarshmallow4Warning
def ensure_text_type(val: str | bytes) -> str:
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    return str(val)