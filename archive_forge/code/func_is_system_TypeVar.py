import sys
import typing
from struct import Struct
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx.deprecation import RemovedInSphinx60Warning, deprecated_alias
def is_system_TypeVar(typ: Any) -> bool:
    """Check *typ* is system defined TypeVar."""
    modname = getattr(typ, '__module__', '')
    return modname == 'typing' and isinstance(typ, TypeVar)