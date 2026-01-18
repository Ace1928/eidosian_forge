import time
from collections import OrderedDict as _OrderedDict
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping, MutableSet, Sequence
from heapq import heapify, heappop, heappush
from itertools import chain, count
from queue import Empty
from typing import Any, Dict, Iterable, List  # noqa
from .functional import first, uniq
from .text import match_case
def lpmerge(L, R):
    """In place left precedent dictionary merge.

    Keeps values from `L`, if the value in `R` is :const:`None`.
    """
    setitem = L.__setitem__
    [setitem(k, v) for k, v in R.items() if v is not None]
    return L