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
def check_for_b907(self, node: ast.JoinedStr):

    def myunparse(node: ast.AST) -> str:
        if sys.version_info >= (3, 9):
            return ast.unparse(node)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return myunparse(node.value) + '.' + node.attr
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Call):
            return myunparse(node.func) + '()'
        return type(node).__name__
    quote_marks = '\'"'
    current_mark = None
    variable = None
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if not value.value:
                continue
            if current_mark is not None and variable is not None and (value.value[0] == current_mark):
                self.errors.append(B907(variable.lineno, variable.col_offset, vars=(myunparse(variable.value),)))
                current_mark = variable = None
                if len(value.value) == 1:
                    continue
            if value.value[-1] in quote_marks:
                current_mark = value.value[-1]
                variable = None
                continue
        if current_mark is not None and variable is None and isinstance(value, ast.FormattedValue) and (value.conversion != ord('r')):
            if isinstance(value.format_spec, ast.JoinedStr) and value.format_spec.values:
                if len(value.format_spec.values) > 1 or not isinstance(value.format_spec.values[0], ast.Constant):
                    current_mark = variable = None
                    continue
                format_specifier = value.format_spec.values[0].value
                if len(format_specifier) > 1 and format_specifier[1] in '<>=^':
                    format_specifier = format_specifier[1:]
                format_specifier = re.sub('\\.\\d*', '', format_specifier)
                invalid_characters = ''.join(('=', '+- ', '0123456789', 'z', '#', '_,', 'bcdeEfFgGnoxX%'))
                if set(format_specifier).intersection(invalid_characters):
                    current_mark = variable = None
                    continue
            variable = value
            continue
        current_mark = variable = None