from __future__ import annotations
from collections.abc import Iterable
from contextlib import contextmanager
import logging
import sys
import textwrap
from typing import Iterator
from typing import Optional
from typing import TextIO
from typing import Union
import warnings
from sqlalchemy.engine import url
from . import sqla_compat
def format_as_comma(value: Optional[Union[str, Iterable[str]]]) -> str:
    if value is None:
        return ''
    elif isinstance(value, str):
        return value
    elif isinstance(value, Iterable):
        return ', '.join(value)
    else:
        raise ValueError("Don't know how to comma-format %r" % value)