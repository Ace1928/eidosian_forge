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
def getorigbases(obj: Any) -> Optional[Tuple[Any, ...]]:
    """Get __orig_bases__ from *obj* safely."""
    if not inspect.isclass(obj):
        return None
    __dict__ = safe_getattr(obj, '__dict__', {})
    __orig_bases__ = __dict__.get('__orig_bases__')
    if isinstance(__orig_bases__, tuple) and len(__orig_bases__) > 0:
        return __orig_bases__
    else:
        return None