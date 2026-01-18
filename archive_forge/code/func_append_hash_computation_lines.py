import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def append_hash_computation_lines(prefix, indent):
    """
        Generate the code for actually computing the hash code.
        Below this will either be returned directly or used to compute
        a value which is then cached, depending on the value of cache_hash
        """
    method_lines.extend([indent + prefix + hash_func, indent + f'        {type_hash},'])
    for a in attrs:
        if a.eq_key:
            cmp_name = f'_{a.name}_key'
            globs[cmp_name] = a.eq_key
            method_lines.append(indent + f'        {cmp_name}(self.{a.name}),')
        else:
            method_lines.append(indent + f'        self.{a.name},')
    method_lines.append(indent + '    ' + closing_braces)