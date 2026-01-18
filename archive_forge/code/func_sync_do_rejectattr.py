import math
import random
import re
import typing
import typing as t
from collections import abc
from itertools import chain
from itertools import groupby
from markupsafe import escape
from markupsafe import Markup
from markupsafe import soft_str
from .async_utils import async_variant
from .async_utils import auto_aiter
from .async_utils import auto_await
from .async_utils import auto_to_list
from .exceptions import FilterArgumentError
from .runtime import Undefined
from .utils import htmlsafe_json_dumps
from .utils import pass_context
from .utils import pass_environment
from .utils import pass_eval_context
from .utils import pformat
from .utils import url_quote
from .utils import urlize
@pass_context
def sync_do_rejectattr(context: 'Context', value: 't.Iterable[V]', *args: t.Any, **kwargs: t.Any) -> 't.Iterator[V]':
    """Filters a sequence of objects by applying a test to the specified
    attribute of each object, and rejecting the objects with the test
    succeeding.

    If no test is specified, the attribute's value will be evaluated as
    a boolean.

    .. sourcecode:: jinja

        {{ users|rejectattr("is_active") }}
        {{ users|rejectattr("email", "none") }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (u for user in users if not user.is_active)
        (u for user in users if not test_none(user.email))

    .. versionadded:: 2.7
    """
    return select_or_reject(context, value, args, kwargs, lambda x: not x, True)