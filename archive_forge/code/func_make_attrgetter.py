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
def make_attrgetter(environment: 'Environment', attribute: t.Optional[t.Union[str, int]], postprocess: t.Optional[t.Callable[[t.Any], t.Any]]=None, default: t.Optional[t.Any]=None) -> t.Callable[[t.Any], t.Any]:
    """Returns a callable that looks up the given attribute from a
    passed object with the rules of the environment.  Dots are allowed
    to access attributes of attributes.  Integer parts in paths are
    looked up as integers.
    """
    parts = _prepare_attribute_parts(attribute)

    def attrgetter(item: t.Any) -> t.Any:
        for part in parts:
            item = environment.getitem(item, part)
            if default is not None and isinstance(item, Undefined):
                item = default
        if postprocess is not None:
            item = postprocess(item)
        return item
    return attrgetter