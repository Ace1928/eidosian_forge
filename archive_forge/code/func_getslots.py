import builtins
import contextlib
import enum
import inspect
import re
import sys
import types
import typing
from functools import partial, partialmethod
from importlib import import_module
from inspect import Parameter, isclass, ismethod, ismethoddescriptor, ismodule  # NOQA
from io import StringIO
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, cast
from sphinx.pycode.ast import ast  # for py36-37
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import ForwardRef
from sphinx.util.typing import stringify as stringify_annotation
def getslots(obj: Any) -> Optional[Dict]:
    """Get __slots__ attribute of the class as dict.

    Return None if gienv *obj* does not have __slots__.
    Raises TypeError if given *obj* is not a class.
    Raises ValueError if given *obj* have invalid __slots__.
    """
    if not inspect.isclass(obj):
        raise TypeError
    __slots__ = safe_getattr(obj, '__slots__', None)
    if __slots__ is None:
        return None
    elif isinstance(__slots__, dict):
        return __slots__
    elif isinstance(__slots__, str):
        return {__slots__: None}
    elif isinstance(__slots__, (list, tuple)):
        return {e: None for e in __slots__}
    else:
        raise ValueError