import typing as t
from contextlib import contextmanager
from functools import update_wrapper
from io import StringIO
from itertools import chain
from keyword import iskeyword as is_python_keyword
from markupsafe import escape
from markupsafe import Markup
from . import nodes
from .exceptions import TemplateAssertionError
from .idtracking import Symbols
from .idtracking import VAR_LOAD_ALIAS
from .idtracking import VAR_LOAD_PARAMETER
from .idtracking import VAR_LOAD_RESOLVE
from .idtracking import VAR_LOAD_UNDEFINED
from .nodes import EvalContext
from .optimizer import Optimizer
from .utils import _PassArg
from .utils import concat
from .visitor import NodeVisitor
def has_safe_repr(value: t.Any) -> bool:
    """Does the node have a safe representation?"""
    if value is None or value is NotImplemented or value is Ellipsis:
        return True
    if type(value) in {bool, int, float, complex, range, str, Markup}:
        return True
    if type(value) in {tuple, list, set, frozenset}:
        return all((has_safe_repr(v) for v in value))
    if type(value) is dict:
        return all((has_safe_repr(k) and has_safe_repr(v) for k, v in value.items()))
    return False