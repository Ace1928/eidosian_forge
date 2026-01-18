from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
@classmethod
def from_groups(cls, *groups):
    return Alphabet({s: TransitionKey(i) for i, group in enumerate(groups) for s in group})