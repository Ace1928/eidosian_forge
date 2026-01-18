import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
def classify(seq: Iterable, key: Optional[Callable]=None, value: Optional[Callable]=None) -> Dict:
    d: Dict[Any, Any] = {}
    for item in seq:
        k = key(item) if key is not None else item
        v = value(item) if value is not None else item
        try:
            d[k].append(v)
        except KeyError:
            d[k] = [v]
    return d