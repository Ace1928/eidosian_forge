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
def make_multi_attrgetter(environment: 'Environment', attribute: t.Optional[t.Union[str, int]], postprocess: t.Optional[t.Callable[[t.Any], t.Any]]=None) -> t.Callable[[t.Any], t.List[t.Any]]:
    """Returns a callable that looks up the given comma separated
    attributes from a passed object with the rules of the environment.
    Dots are allowed to access attributes of each attribute.  Integer
    parts in paths are looked up as integers.

    The value returned by the returned callable is a list of extracted
    attribute values.

    Examples of attribute: "attr1,attr2", "attr1.inner1.0,attr2.inner2.0", etc.
    """
    if isinstance(attribute, str):
        split: t.Sequence[t.Union[str, int, None]] = attribute.split(',')
    else:
        split = [attribute]
    parts = [_prepare_attribute_parts(item) for item in split]

    def attrgetter(item: t.Any) -> t.List[t.Any]:
        items = [None] * len(parts)
        for i, attribute_part in enumerate(parts):
            item_i = item
            for part in attribute_part:
                item_i = environment.getitem(item_i, part)
            if postprocess is not None:
                item_i = postprocess(item_i)
            items[i] = item_i
        return items
    return attrgetter