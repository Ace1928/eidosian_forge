from enum import Enum
from functools import total_ordering
from typing import Literal
from typing import Optional
def next_lower(self) -> 'Scope':
    """Return the next lower scope."""
    index = _SCOPE_INDICES[self]
    if index == 0:
        raise ValueError(f'{self} is the lower-most scope')
    return _ALL_SCOPES[index - 1]