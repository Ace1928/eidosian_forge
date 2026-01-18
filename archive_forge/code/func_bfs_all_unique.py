import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
def bfs_all_unique(initial, expand):
    """bfs, but doesn't keep track of visited (aka seen), because there can be no repetitions"""
    open_q = deque(list(initial))
    while open_q:
        node = open_q.popleft()
        yield node
        open_q += expand(node)