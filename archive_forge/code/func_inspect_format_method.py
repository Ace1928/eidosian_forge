import operator
import types
import typing as t
from _string import formatter_field_name_split  # type: ignore
from collections import abc
from collections import deque
from string import Formatter
from markupsafe import EscapeFormatter
from markupsafe import Markup
from .environment import Environment
from .exceptions import SecurityError
from .runtime import Context
from .runtime import Undefined
def inspect_format_method(callable: t.Callable) -> t.Optional[str]:
    if not isinstance(callable, (types.MethodType, types.BuiltinMethodType)) or callable.__name__ not in ('format', 'format_map'):
        return None
    obj = callable.__self__
    if isinstance(obj, str):
        return obj
    return None