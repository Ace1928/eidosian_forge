import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
def combine_alternatives(lists):
    """
    Accepts a list of alternatives, and enumerates all their possible concatenations.

    Examples:
        >>> combine_alternatives([range(2), [4,5]])
        [[0, 4], [0, 5], [1, 4], [1, 5]]

        >>> combine_alternatives(["abc", "xy", '$'])
        [['a', 'x', '$'], ['a', 'y', '$'], ['b', 'x', '$'], ['b', 'y', '$'], ['c', 'x', '$'], ['c', 'y', '$']]

        >>> combine_alternatives([])
        [[]]
    """
    if not lists:
        return [[]]
    assert all((l for l in lists)), lists
    return list(product(*lists))