from __future__ import annotations
from inspect import getfullargspec
def normalize_repr(v):
    """
    Return dictionary repr sorted by keys, leave others unchanged

    >>> normalize_repr({1:2,3:4,5:6,7:8})
    '{1: 2, 3: 4, 5: 6, 7: 8}'
    >>> normalize_repr('foo')
    "'foo'"
    """
    if isinstance(v, dict):
        items = sorted(((repr(k), repr(v)) for k, v in v.items()))
        return f'{{{', '.join((f'{k}: {v}' for k, v in items))}}}'
    return repr(v)