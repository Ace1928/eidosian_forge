import itertools
import heapq
import collections
import operator
from functools import partial
from itertools import filterfalse, zip_longest
from collections.abc import Sequence
from toolz.utils import no_default
def peekn(n, seq):
    """ Retrieve the next n elements of a sequence

    Returns a tuple of the first n elements and an iterable equivalent
    to the original, still having the elements retrieved.

    >>> seq = [0, 1, 2, 3, 4]
    >>> first_two, seq = peekn(2, seq)
    >>> first_two
    (0, 1)
    >>> list(seq)
    [0, 1, 2, 3, 4]
    """
    iterator = iter(seq)
    peeked = tuple(take(n, iterator))
    return (peeked, itertools.chain(iter(peeked), iterator))