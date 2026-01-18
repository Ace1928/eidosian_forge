from __future__ import annotations
from .common import CMakeException
from .generator import parse_generator_expressions
from .. import mlog
from ..mesonlib import version_compare
import typing as T
from pathlib import Path
from functools import lru_cache
import re
import json
import textwrap
def handle_working_dir(key: str, target: CMakeGeneratorTarget) -> None:
    nonlocal working_dir
    if working_dir is None:
        working_dir = key
    else:
        working_dir += ' '
        working_dir += key