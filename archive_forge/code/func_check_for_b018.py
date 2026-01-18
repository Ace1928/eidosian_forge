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
def check_for_b018(self, node):
    if not isinstance(node, ast.Expr):
        return
    if isinstance(node.value, (ast.List, ast.Set, ast.Dict, ast.Tuple)) or (isinstance(node.value, ast.Constant) and (isinstance(node.value.value, (int, float, complex, bytes, bool)) or node.value.value is None)):
        self.errors.append(B018(node.lineno, node.col_offset, vars=(node.value.__class__.__name__,)))