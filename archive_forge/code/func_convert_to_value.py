import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def convert_to_value(item):
    if isinstance(item, ast.Constant):
        return item.value
    elif isinstance(item, ast.Tuple):
        return tuple((convert_to_value(i) for i in item.elts))
    elif isinstance(item, ast.Name):
        return VariableKey(item=item)
    else:
        return UnhandledKeyType()