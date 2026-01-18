import inspect
import types
import traceback
import sys
import operator as op
from collections import namedtuple
import warnings
import re
import builtins
import typing
from pathlib import Path
from typing import Optional, Tuple
from jedi.inference.compiled.getattr_static import getattr_static
def getattr_paths(self, name, default=_sentinel):
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            return_obj = getattr(self._obj, name)
    except Exception as e:
        if default is _sentinel:
            if isinstance(e, AttributeError):
                raise
            raise AttributeError
        return_obj = default
    access = self._create_access(return_obj)
    if inspect.ismodule(return_obj):
        return [access]
    try:
        module = return_obj.__module__
    except AttributeError:
        pass
    else:
        if module is not None and isinstance(module, str):
            try:
                __import__(module)
            except ImportError:
                pass
    module = inspect.getmodule(return_obj)
    if module is None:
        module = inspect.getmodule(type(return_obj))
        if module is None:
            module = builtins
    return [self._create_access(module), access]