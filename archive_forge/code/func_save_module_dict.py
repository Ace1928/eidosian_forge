from __future__ import annotations
import warnings
from .logger import adapter as logger
from .logger import trace as _trace
import os
import sys
import builtins as __builtin__
from pickle import _Pickler as StockPickler, Unpickler as StockUnpickler
from pickle import GLOBAL, POP
from _thread import LockType
from _thread import RLock as RLockType
from types import CodeType, FunctionType, MethodType, GeneratorType, \
from types import MappingProxyType as DictProxyType, new_class
from pickle import DEFAULT_PROTOCOL, HIGHEST_PROTOCOL, PickleError, PicklingError, UnpicklingError
import __main__ as _main_module
import marshal
import gc
import abc
import dataclasses
from weakref import ReferenceType, ProxyType, CallableProxyType
from collections import OrderedDict
from enum import Enum, EnumMeta
from functools import partial
from operator import itemgetter, attrgetter
import importlib.machinery
from types import GetSetDescriptorType, ClassMethodDescriptorType, \
from io import BytesIO as StringIO
from socket import socket as SocketType
from multiprocessing.reduction import _reduce_socket as reduce_socket
import inspect
import typing
from . import _shims
from ._shims import Reduce, Getattr
@register(dict)
def save_module_dict(pickler, obj):
    if is_dill(pickler, child=False) and obj == pickler._main.__dict__ and (not (pickler._session and pickler._first_pass)):
        logger.trace(pickler, 'D1: %s', _repr_dict(obj))
        pickler.write(bytes('c__builtin__\n__main__\n', 'UTF-8'))
        logger.trace(pickler, '# D1')
    elif not is_dill(pickler, child=False) and obj == _main_module.__dict__:
        logger.trace(pickler, 'D3: %s', _repr_dict(obj))
        pickler.write(bytes('c__main__\n__dict__\n', 'UTF-8'))
        logger.trace(pickler, '# D3')
    elif '__name__' in obj and obj != _main_module.__dict__ and (type(obj['__name__']) is str) and (obj is getattr(_import_module(obj['__name__'], True), '__dict__', None)):
        logger.trace(pickler, 'D4: %s', _repr_dict(obj))
        pickler.write(bytes('c%s\n__dict__\n' % obj['__name__'], 'UTF-8'))
        logger.trace(pickler, '# D4')
    else:
        logger.trace(pickler, 'D2: %s', _repr_dict(obj))
        if is_dill(pickler, child=False) and pickler._session:
            pickler._first_pass = False
        StockPickler.save_dict(pickler, obj)
        logger.trace(pickler, '# D2')
    return