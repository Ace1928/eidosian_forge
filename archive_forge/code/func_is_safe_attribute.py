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
def is_safe_attribute(self, obj: t.Any, attr: str, value: t.Any) -> bool:
    if not super().is_safe_attribute(obj, attr, value):
        return False
    return not modifies_known_mutable(obj, attr)