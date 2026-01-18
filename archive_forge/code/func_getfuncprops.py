import inspect
import keyword
import pydoc
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Dict, List, ContextManager
from types import MemberDescriptorType, TracebackType
from ._typing_compat import Literal
from pygments.token import Token
from pygments.lexers import Python3Lexer
from .lazyre import LazyReCompile
def getfuncprops(func: str, f: Callable) -> Optional[FuncProps]:
    try:
        func_name = getattr(f, '__name__', None)
    except:
        func_name = None
    try:
        is_bound_method = inspect.ismethod(f) and f.__self__ is not None or (func_name == '__init__' and (not func.endswith('.__init__'))) or (func_name == '__new__' and (not func.endswith('.__new__')))
    except:
        return None
    try:
        argspec = _get_argspec_from_signature(f)
        fprops = FuncProps(func, _fix_default_values(f, argspec), is_bound_method)
    except (TypeError, KeyError, ValueError):
        argspec_pydoc = _getpydocspec(f)
        if argspec_pydoc is None:
            return None
        if inspect.ismethoddescriptor(f):
            argspec_pydoc.args.insert(0, 'obj')
        fprops = FuncProps(func, argspec_pydoc, is_bound_method)
    return fprops