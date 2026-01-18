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
def prepare_map(context: 'Context', args: t.Tuple, kwargs: t.Dict[str, t.Any]) -> t.Callable[[t.Any], t.Any]:
    if not args and 'attribute' in kwargs:
        attribute = kwargs.pop('attribute')
        default = kwargs.pop('default', None)
        if kwargs:
            raise FilterArgumentError(f'Unexpected keyword argument {next(iter(kwargs))!r}')
        func = make_attrgetter(context.environment, attribute, default=default)
    else:
        try:
            name = args[0]
            args = args[1:]
        except LookupError:
            raise FilterArgumentError('map requires a filter argument') from None

        def func(item: t.Any) -> t.Any:
            return context.environment.call_filter(name, item, args, kwargs, context=context)
    return func