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
def get_first_cmake_var_of(self, var_list: T.List[str]) -> T.List[str]:
    for i in var_list:
        if i in self.vars:
            return self.vars[i]
    return []