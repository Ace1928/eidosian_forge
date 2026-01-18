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
def fmt_setter(attr_name, value_var, has_on_setattr):
    if _is_slot_attr(attr_name, base_attr_map):
        return _setattr(attr_name, value_var, has_on_setattr)
    return f"_inst_dict['{attr_name}'] = {value_var}"