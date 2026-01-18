import os
from . import _compat
def mapping_items(mapping, _iteritems=_compat.iteritems):
    """Return an iterator over the ``mapping`` items, sort if it's a plain dict.

    >>> list(mapping_items({'spam': 0, 'ham': 1, 'eggs': 2}))
    [('eggs', 2), ('ham', 1), ('spam', 0)]

    >>> from collections import OrderedDict
    >>> list(mapping_items(OrderedDict(enumerate(['spam', 'ham', 'eggs']))))
    [(0, 'spam'), (1, 'ham'), (2, 'eggs')]
    """
    if type(mapping) is dict:
        return iter(sorted(_iteritems(mapping)))
    return _iteritems(mapping)