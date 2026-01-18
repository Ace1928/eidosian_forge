from __future__ import annotations
import ast
import io
import os
import sys
import tokenize
from collections.abc import (
from os.path import relpath
from textwrap import dedent
from tokenize import COMMENT, NAME, OP, STRING, generate_tokens
from typing import TYPE_CHECKING, Any
from babel.util import parse_encoding, parse_future_flags, pathmatch
def default_directory_filter(dirpath: str | os.PathLike[str]) -> bool:
    subdir = os.path.basename(dirpath)
    return not (subdir.startswith('.') or subdir.startswith('_'))