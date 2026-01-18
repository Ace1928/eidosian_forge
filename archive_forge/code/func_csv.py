from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
def csv(*args: str, sep: str=', ') -> str:
    """
    Formats any number of string arguments as CSV.

    Args:
        args: The string arguments to format.
        sep: The argument separator.

    Returns:
        The arguments formatted as a CSV string.
    """
    return sep.join((arg for arg in args if arg))