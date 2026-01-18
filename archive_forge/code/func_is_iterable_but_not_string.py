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
def is_iterable_but_not_string(obj) -> bool:
    """Return True if ``obj`` is an iterable object that isn't a string."""
    return hasattr(obj, '__iter__') and (not hasattr(obj, 'strip')) or is_generator(obj)