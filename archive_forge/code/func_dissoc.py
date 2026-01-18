import collections
import operator
from functools import reduce
from collections.abc import Mapping
def dissoc(d, *keys, **kwargs):
    """ Return a new dict with the given key(s) removed.

    New dict has d[key] deleted for each supplied key.
    Does not modify the initial dictionary.

    >>> dissoc({'x': 1, 'y': 2}, 'y')
    {'x': 1}
    >>> dissoc({'x': 1, 'y': 2}, 'y', 'x')
    {}
    >>> dissoc({'x': 1}, 'y') # Ignores missing keys
    {'x': 1}
    """
    factory = _get_factory(dissoc, kwargs)
    d2 = factory()
    if len(keys) < len(d) * 0.6:
        d2.update(d)
        for key in keys:
            if key in d2:
                del d2[key]
    else:
        remaining = set(d)
        remaining.difference_update(keys)
        for k in remaining:
            d2[k] = d[k]
    return d2