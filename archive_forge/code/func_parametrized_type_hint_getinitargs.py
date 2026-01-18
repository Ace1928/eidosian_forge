import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
def parametrized_type_hint_getinitargs(obj):
    if type(obj) is type(Literal):
        initargs = (Literal, obj.__values__)
    elif type(obj) is type(Final):
        initargs = (Final, obj.__type__)
    elif type(obj) is type(ClassVar):
        initargs = (ClassVar, obj.__type__)
    elif type(obj) is type(Generic):
        initargs = (obj.__origin__, obj.__args__)
    elif type(obj) is type(Union):
        initargs = (Union, obj.__args__)
    elif type(obj) is type(Tuple):
        initargs = (Tuple, obj.__args__)
    elif type(obj) is type(Callable):
        *args, result = obj.__args__
        if len(args) == 1 and args[0] is Ellipsis:
            args = Ellipsis
        else:
            args = list(args)
        initargs = (Callable, (args, result))
    else:
        raise pickle.PicklingError(f'Cloudpickle Error: Unknown type {type(obj)}')
    return initargs