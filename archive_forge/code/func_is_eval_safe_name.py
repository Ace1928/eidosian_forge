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
def is_eval_safe_name(string: str) -> bool:
    return all((part.isidentifier() and (not keyword.iskeyword(part)) for part in string.split('.')))