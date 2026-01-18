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
def check_for_b901(self, node):
    if node.name == '__await__':
        return
    has_yield = False
    return_node = None
    for parent, x in self.walk_function_body(node):
        if isinstance(parent, ast.Expr) and isinstance(x, (ast.Yield, ast.YieldFrom)):
            has_yield = True
        if isinstance(x, ast.Return) and x.value is not None:
            return_node = x
        if has_yield and return_node is not None:
            self.errors.append(B901(return_node.lineno, return_node.col_offset))
            break