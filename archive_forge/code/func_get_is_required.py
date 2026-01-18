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
@staticmethod
def get_is_required(cls: ClassDef, stmt: AssignmentStmt, lhs: NameExpr) -> bool:
    """
        Returns a boolean indicating whether the field defined in `stmt` is a required field.
        """
    expr = stmt.rvalue
    if isinstance(expr, TempNode):
        value_type = get_proper_type(cls.info[lhs.name].type)
        return not PydanticModelTransformer.type_has_implicit_default(value_type)
    if isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr) and (expr.callee.fullname == FIELD_FULLNAME):
        for arg, name in zip(expr.args, expr.arg_names):
            if name is None or name == 'default':
                return arg.__class__ is EllipsisExpr
            if name == 'default_factory':
                return False
        value_type = get_proper_type(cls.info[lhs.name].type)
        return not PydanticModelTransformer.type_has_implicit_default(value_type)
    return isinstance(expr, EllipsisExpr)