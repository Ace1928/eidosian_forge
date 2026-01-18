import functools
import sys
import typing as t
from collections import abc
from itertools import chain
from markupsafe import escape  # noqa: F401
from markupsafe import Markup
from markupsafe import soft_str
from .async_utils import auto_aiter
from .async_utils import auto_await  # noqa: F401
from .exceptions import TemplateNotFound  # noqa: F401
from .exceptions import TemplateRuntimeError  # noqa: F401
from .exceptions import UndefinedError
from .nodes import EvalContext
from .utils import _PassArg
from .utils import concat
from .utils import internalcode
from .utils import missing
from .utils import Namespace  # noqa: F401
from .utils import object_type_repr
from .utils import pass_eval_context
@property
def nextitem(self) -> t.Union[t.Any, 'Undefined']:
    """The item in the next iteration. Undefined during the last
        iteration.

        Causes the iterable to advance early. See
        :func:`itertools.groupby` for issues this can cause.
        The :func:`jinja-filters.groupby` filter avoids that issue.
        """
    rv = self._peek_next()
    if rv is missing:
        return self._undefined('there is no next item')
    return rv