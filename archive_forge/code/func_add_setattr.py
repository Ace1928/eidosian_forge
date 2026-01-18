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
def add_setattr(self):
    if self._frozen:
        return self
    sa_attrs = {}
    for a in self._attrs:
        on_setattr = a.on_setattr or self._on_setattr
        if on_setattr and on_setattr is not setters.NO_OP:
            sa_attrs[a.name] = (a, on_setattr)
    if not sa_attrs:
        return self
    if self._has_custom_setattr:
        msg = "Can't combine custom __setattr__ with on_setattr hooks."
        raise ValueError(msg)

    def __setattr__(self, name, val):
        try:
            a, hook = sa_attrs[name]
        except KeyError:
            nval = val
        else:
            nval = hook(self, a, val)
        _obj_setattr(self, name, nval)
    self._cls_dict['__attrs_own_setattr__'] = True
    self._cls_dict['__setattr__'] = self._add_method_dunders(__setattr__)
    self._wrote_own_setattr = True
    return self