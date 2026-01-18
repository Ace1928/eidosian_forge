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
def do_default(value: V, default_value: V='', boolean: bool=False) -> V:
    """If the value is undefined it will return the passed default value,
    otherwise the value of the variable:

    .. sourcecode:: jinja

        {{ my_variable|default('my_variable is not defined') }}

    This will output the value of ``my_variable`` if the variable was
    defined, otherwise ``'my_variable is not defined'``. If you want
    to use default with variables that evaluate to false you have to
    set the second parameter to `true`:

    .. sourcecode:: jinja

        {{ ''|default('the string was empty', true) }}

    .. versionchanged:: 2.11
       It's now possible to configure the :class:`~jinja2.Environment` with
       :class:`~jinja2.ChainableUndefined` to make the `default` filter work
       on nested elements and attributes that may contain undefined values
       in the chain without getting an :exc:`~jinja2.UndefinedError`.
    """
    if isinstance(value, Undefined) or (boolean and (not value)):
        return default_value
    return value