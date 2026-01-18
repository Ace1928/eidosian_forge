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
def getattr_safe(obj: Any, name: str) -> Any:
    """Side effect free getattr (calls getattr_static)."""
    result = inspect.getattr_static(obj, name)
    if isinstance(result, MemberDescriptorType):
        result = getattr(obj, name)
    if isinstance(result, (classmethod, staticmethod)):
        result = result.__get__(obj, obj)
    return result