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
@register(LRUCacheType)
def save_lru_cache(pickler, obj):
    logger.trace(pickler, 'LRU: %s', obj)
    if OLD39:
        kwargs = obj.cache_info()
        args = (kwargs.maxsize,)
    else:
        kwargs = obj.cache_parameters()
        args = (kwargs['maxsize'], kwargs['typed'])
    if args != lru_cache.__defaults__:
        wrapper = Reduce(lru_cache, args, is_callable=True)
    else:
        wrapper = lru_cache
    pickler.save_reduce(wrapper, (obj.__wrapped__,), obj=obj)
    logger.trace(pickler, '# LRU')
    return