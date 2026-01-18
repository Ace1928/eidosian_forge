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
@register(PyCapsuleType)
def save_capsule(pickler, obj):
    logger.trace(pickler, 'Cap: %s', obj)
    name = _PyCapsule_GetName(obj)
    pointer = _PyCapsule_GetPointer(obj, name)
    context = _PyCapsule_GetContext(obj)
    destructor = _PyCapsule_GetDestructor(obj)
    pickler.save_reduce(_create_capsule, (pointer, name, context, destructor), obj=obj)
    logger.trace(pickler, '# Cap')