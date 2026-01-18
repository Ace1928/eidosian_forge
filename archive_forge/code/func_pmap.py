from collections.abc import Mapping, Hashable
from itertools import chain
from typing import Generic, TypeVar
from pyrsistent._pvector import pvector
from pyrsistent._transformations import transform
def pmap(initial={}, pre_size=0):
    """
    Create new persistent map, inserts all elements in initial into the newly created map.
    The optional argument pre_size may be used to specify an initial size of the underlying bucket vector. This
    may have a positive performance impact in the cases where you know beforehand that a large number of elements
    will be inserted into the map eventually since it will reduce the number of reallocations required.

    >>> pmap({'a': 13, 'b': 14}) == {'a': 13, 'b': 14}
    True
    """
    if not initial and pre_size == 0:
        return _EMPTY_PMAP
    return _turbo_mapping(initial, pre_size)