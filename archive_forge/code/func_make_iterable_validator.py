import locale
import logging
import os
import pprint
import re
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Literal
import numpy as np
def make_iterable_validator(scalar_validator, length=None, allow_none=False, allow_auto=False):
    """Validate value is an iterable datatype."""

    def validate_iterable(value):
        if allow_none and (value is None or (isinstance(value, str) and value.lower() == 'none')):
            return None
        if isinstance(value, str):
            if allow_auto and value.lower() == 'auto':
                return 'auto'
            value = tuple((v.strip('([ ])') for v in value.split(',') if v.strip()))
        if np.iterable(value) and (not isinstance(value, (set, frozenset))):
            val = tuple((scalar_validator(v) for v in value))
            if length is not None and len(val) != length:
                raise ValueError(f'Iterable must be of length: {length}')
            return val
        raise ValueError('Only ordered iterable values are valid')
    return validate_iterable