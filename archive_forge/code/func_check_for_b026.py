from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
def check_for_b026(self, call: ast.Call):
    if not call.keywords:
        return
    starreds = [arg for arg in call.args if isinstance(arg, ast.Starred)]
    if not starreds:
        return
    first_keyword = call.keywords[0].value
    for starred in starreds:
        if (starred.lineno, starred.col_offset) > (first_keyword.lineno, first_keyword.col_offset):
            self.errors.append(B026(starred.lineno, starred.col_offset))