import sys
import typing
from struct import Struct
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx.deprecation import RemovedInSphinx60Warning, deprecated_alias
def is_invalid_builtin_class(obj: Any) -> bool:
    """Check *obj* is an invalid built-in class."""
    try:
        return obj in INVALID_BUILTIN_CLASSES
    except TypeError:
        return False