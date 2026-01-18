import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
def dedup_list(l: Sequence[T]) -> List[T]:
    """Given a list (l) will removing duplicates from the list,
       preserving the original order of the list. Assumes that
       the list entries are hashable."""
    dedup = set()
    return [x for x in l if not (x in dedup or dedup.add(x))]