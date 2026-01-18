import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type as TypingType, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name  # type: ignore
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic.utils import is_valid_field
def adjust_validator_signatures(self) -> None:
    """When we decorate a function `f` with `pydantic.validator(...), mypy sees
        `f` as a regular method taking a `self` instance, even though pydantic
        internally wraps `f` with `classmethod` if necessary.

        Teach mypy this by marking any function whose outermost decorator is a
        `validator()` call as a classmethod.
        """
    for name, sym in self._ctx.cls.info.names.items():
        if isinstance(sym.node, Decorator):
            first_dec = sym.node.original_decorators[0]
            if isinstance(first_dec, CallExpr) and isinstance(first_dec.callee, NameExpr) and (first_dec.callee.fullname == f'{_NAMESPACE}.class_validators.validator'):
                sym.node.func.is_class = True