from __future__ import absolute_import, print_function, division
import operator
from functools import partial
from petl.compat import text_type, binary_type, numeric_types
def comparable_itemgetter(*args):
    getter = operator.itemgetter(*args)
    getter_with_default = _itemgetter_with_default(*args)

    def _getter_with_fallback(obj):
        try:
            return getter(obj)
        except (IndexError, KeyError):
            return getter_with_default(obj)
    g = lambda x: Comparable(_getter_with_fallback(x))
    return g