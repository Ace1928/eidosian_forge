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
def stdin_get_lines() -> list[str]:
    """Return lines of stdin split according to file splitting."""
    return list(io.StringIO(stdin_get_value()))