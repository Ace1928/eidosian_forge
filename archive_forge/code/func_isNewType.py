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
def isNewType(obj: Any) -> bool:
    """Check the if object is a kind of NewType."""
    if sys.version_info >= (3, 10):
        return isinstance(obj, typing.NewType)
    else:
        __module__ = safe_getattr(obj, '__module__', None)
        __qualname__ = safe_getattr(obj, '__qualname__', None)
        if __module__ == 'typing' and __qualname__ == 'NewType.<locals>.new_type':
            return True
        else:
            return False