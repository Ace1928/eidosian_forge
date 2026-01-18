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
def hasattr_safe(obj: Any, name: str) -> bool:
    try:
        getattr_safe(obj, name)
        return True
    except AttributeError:
        return False