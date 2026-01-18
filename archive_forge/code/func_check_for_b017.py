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
def check_for_b017(self, node):
    """Checks for use of the evil syntax 'with assertRaises(Exception):'
        or 'with pytest.raises(Exception)'.

        This form of assertRaises will catch everything that subclasses
        Exception, which happens to be the vast majority of Python internal
        errors, including the ones raised when a non-existing method/function
        is called, or a function is called with an invalid dictionary key
        lookup.
        """
    item = node.items[0]
    item_context = item.context_expr
    if hasattr(item_context, 'func') and (isinstance(item_context.func, ast.Attribute) and (item_context.func.attr == 'assertRaises' or (item_context.func.attr == 'raises' and isinstance(item_context.func.value, ast.Name) and (item_context.func.value.id == 'pytest') and ('match' not in (kwd.arg for kwd in item_context.keywords)))) or (isinstance(item_context.func, ast.Name) and item_context.func.id == 'raises' and isinstance(item_context.func.ctx, ast.Load) and ('pytest.raises' in self._b005_imports) and ('match' not in (kwd.arg for kwd in item_context.keywords)))) and (len(item_context.args) == 1) and isinstance(item_context.args[0], ast.Name) and (item_context.args[0].id in {'Exception', 'BaseException'}) and (not item.optional_vars):
        self.errors.append(B017(node.lineno, node.col_offset))