from enum import IntEnum
from functools import lru_cache
from itertools import filterfalse
from logging import getLogger
from operator import attrgetter
from typing import (
from .cells import (
from .repr import Result, rich_repr
from .style import Style
@property
def cell_length(self) -> int:
    """The number of terminal cells required to display self.text.

        Returns:
            int: A number of cells.
        """
    text, _style, control = self
    return 0 if control else cell_len(text)