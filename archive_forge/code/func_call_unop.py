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
def call_unop(self, context: Context, operator: str, arg: t.Any) -> t.Any:
    """For intercepted unary operator calls (:meth:`intercepted_unops`)
        this function is executed instead of the builtin operator.  This can
        be used to fine tune the behavior of certain operators.

        .. versionadded:: 2.6
        """
    return self.unop_table[operator](arg)