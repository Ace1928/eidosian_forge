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
def prepare_select_or_reject(context: 'Context', args: t.Tuple, kwargs: t.Dict[str, t.Any], modfunc: t.Callable[[t.Any], t.Any], lookup_attr: bool) -> t.Callable[[t.Any], t.Any]:
    if lookup_attr:
        try:
            attr = args[0]
        except LookupError:
            raise FilterArgumentError('Missing parameter for attribute name') from None
        transfunc = make_attrgetter(context.environment, attr)
        off = 1
    else:
        off = 0

        def transfunc(x: V) -> V:
            return x
    try:
        name = args[off]
        args = args[1 + off:]

        def func(item: t.Any) -> t.Any:
            return context.environment.call_test(name, item, args, kwargs)
    except LookupError:
        func = bool
    return lambda item: modfunc(func(transfunc(item)))