import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def getattr_static(obj, attr, default=_sentinel):
    """Retrieve attributes without triggering dynamic lookup via the
       descriptor protocol,  __getattr__ or __getattribute__.

       Note: this function may not be able to retrieve all attributes
       that getattr can fetch (like dynamically created attributes)
       and may find attributes that getattr can't (like descriptors
       that raise AttributeError). It can also return descriptor objects
       instead of instance members in some cases. See the
       documentation for details.
    """
    instance_result = _sentinel
    if not _is_type(obj):
        klass = type(obj)
        dict_attr = _shadowed_dict(klass)
        if dict_attr is _sentinel or type(dict_attr) is types.MemberDescriptorType:
            instance_result = _check_instance(obj, attr)
    else:
        klass = obj
    klass_result = _check_class(klass, attr)
    if instance_result is not _sentinel and klass_result is not _sentinel:
        if _check_class(type(klass_result), '__get__') is not _sentinel and (_check_class(type(klass_result), '__set__') is not _sentinel or _check_class(type(klass_result), '__delete__') is not _sentinel):
            return klass_result
    if instance_result is not _sentinel:
        return instance_result
    if klass_result is not _sentinel:
        return klass_result
    if obj is klass:
        for entry in _static_getmro(type(klass)):
            if _shadowed_dict(type(entry)) is _sentinel:
                try:
                    return entry.__dict__[attr]
                except KeyError:
                    pass
    if default is not _sentinel:
        return default
    raise AttributeError(attr)