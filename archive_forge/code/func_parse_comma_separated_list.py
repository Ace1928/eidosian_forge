from __future__ import annotations
import fnmatch as _fnmatch
import functools
import io
import logging
import os
import platform
import re
import sys
import textwrap
import tokenize
from typing import NamedTuple
from typing import Pattern
from typing import Sequence
from flake8 import exceptions
def parse_comma_separated_list(value: str, regexp: Pattern[str]=COMMA_SEPARATED_LIST_RE) -> list[str]:
    """Parse a comma-separated list.

    :param value:
        String to be parsed and normalized.
    :param regexp:
        Compiled regular expression used to split the value when it is a
        string.
    :returns:
        List of values with whitespace stripped.
    """
    assert isinstance(value, str), value
    separated = regexp.split(value)
    item_gen = (item.strip() for item in separated)
    return [item for item in item_gen if item]