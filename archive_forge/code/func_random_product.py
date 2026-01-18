import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def random_product(*args, repeat=1):
    """Draw an item at random from each of the input iterables.

        >>> random_product('abc', range(4), 'XYZ')  # doctest:+SKIP
        ('c', 3, 'Z')

    If *repeat* is provided as a keyword argument, that many items will be
    drawn from each iterable.

        >>> random_product('abcd', range(4), repeat=2)  # doctest:+SKIP
        ('a', 2, 'd', 3)

    This equivalent to taking a random selection from
    ``itertools.product(*args, **kwarg)``.

    """
    pools = [tuple(pool) for pool in args] * repeat
    return tuple((choice(pool) for pool in pools))