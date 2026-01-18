from enum import Enum
from functools import total_ordering
from typing import Literal
from typing import Optional
def next_higher(self) -> 'Scope':
    """Return the next higher scope."""
    index = _SCOPE_INDICES[self]
    if index == len(_SCOPE_INDICES) - 1:
        raise ValueError(f'{self} is the upper-most scope')
    return _ALL_SCOPES[index + 1]