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
@classmethod
def from_counting_attr(cls, name, ca, type=None):
    if type is None:
        type = ca.type
    elif ca.type is not None:
        msg = 'Type annotation and type argument cannot both be present'
        raise ValueError(msg)
    inst_dict = {k: getattr(ca, k) for k in Attribute.__slots__ if k not in ('name', 'validator', 'default', 'type', 'inherited')}
    return cls(name=name, validator=ca._validator, default=ca._default, type=type, cmp=None, inherited=False, **inst_dict)