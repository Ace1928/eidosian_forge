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
def swap_with(self, other):
    changes = other.__dict__['changes']
    defaults = other.__dict__['defaults']
    self.__dict__.update(changes=changes, defaults=defaults, key_t=other.__dict__['key_t'], prefix=other.__dict__['prefix'], maps=[changes] + defaults)