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
def normalize_pypi_name(s: str) -> str:
    """Normalize a distribution name according to PEP 503."""
    return NORMALIZE_PACKAGE_NAME_RE.sub('-', s).lower()