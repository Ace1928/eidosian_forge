from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
def make_overload_method_template(typ, attr, overload_func, inline, prefer_literal=False, **kwargs):
    """
    Make a template class for method *attr* of *typ* overloaded by
    *overload_func*.
    """
    return make_overload_attribute_template(typ, attr, overload_func, inline=inline, base=_OverloadMethodTemplate, prefer_literal=prefer_literal, **kwargs)