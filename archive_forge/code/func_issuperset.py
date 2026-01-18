from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def issuperset(self, other):
    """
            Treat `self` and `other` as sets of strings and see if `self` is a
            superset of `other`.
        """
    return (other - self).empty()