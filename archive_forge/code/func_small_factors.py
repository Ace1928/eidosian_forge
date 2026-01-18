import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
def small_factors(n: int, max_factor: int) -> List[Tuple[int, int]]:
    """
    Splits n up into smaller factors and summands <= max_factor.
    Returns a list of [(a, b), ...]
    so that the following code returns n:

    n = 1
    for a, b in values:
        n = n * a + b

    Currently, we also keep a + b <= max_factor, but that might change
    """
    assert n >= 0
    assert max_factor > 2
    if n <= max_factor:
        return [(n, 0)]
    for a in range(max_factor, 1, -1):
        r, b = divmod(n, a)
        if a + b <= max_factor:
            return small_factors(r, max_factor) + [(a, b)]
    assert False, 'Failed to factorize %s' % n