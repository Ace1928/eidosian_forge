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
def fmt_setter_with_converter(attr_name, value_var, has_on_setattr):
    if has_on_setattr or _is_slot_attr(attr_name, base_attr_map):
        return _setattr_with_converter(attr_name, value_var, has_on_setattr)
    return "_inst_dict['%s'] = %s(%s)" % (attr_name, _init_converter_pat % (attr_name,), value_var)