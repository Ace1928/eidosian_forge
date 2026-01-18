from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
def is_constant_expr(self, node: ast.AST) -> str | None:
    """Is this a compile-time constant?"""
    node_name = node.__class__.__name__
    if node_name in ['Constant', 'NameConstant', 'Num']:
        return 'Num'
    elif isinstance(node, ast.Name):
        if node.id in ['True', 'False', 'None', '__debug__']:
            return 'Name'
    return None