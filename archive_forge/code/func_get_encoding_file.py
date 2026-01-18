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
def get_encoding_file(fname: str) -> str:
    """Try to obtain encoding information from a Python source file."""
    with open(fname, encoding='ascii', errors='ignore') as f:
        for _ in range(2):
            line = f.readline()
            match = _get_encoding_line_re.search(line)
            if match:
                return match.group(1)
    return 'utf8'