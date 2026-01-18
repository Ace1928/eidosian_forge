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
def check_for_b903(self, node):
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != '__init__':
        return
    for stmt in body[0].body:
        if not isinstance(stmt, ast.Assign):
            return
        targets = stmt.targets
        if len(targets) > 1 or not isinstance(targets[0], ast.Attribute):
            return
        if not isinstance(stmt.value, ast.Name):
            return
    self.errors.append(B903(node.lineno, node.col_offset))