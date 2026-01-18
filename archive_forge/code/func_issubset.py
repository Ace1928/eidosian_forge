from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def issubset(self, other):
    """
            Treat `self` and `other` as sets of strings and see if `self` is a subset
            of `other`... `self` recognises no strings which `other` doesn't.
        """
    return (self - other).empty()