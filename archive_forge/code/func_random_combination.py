import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def random_combination(iterable, r):
    """Return a random *r* length subsequence of the elements in *iterable*.

        >>> random_combination(range(5), 3)  # doctest:+SKIP
        (2, 3, 4)

    This equivalent to taking a random selection from
    ``itertools.combinations(iterable, r)``.

    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(sample(range(n), r))
    return tuple((pool[i] for i in indices))