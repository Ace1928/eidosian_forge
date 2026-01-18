import collections
import itertools
import re
from typing import Any, Callable, Optional, SupportsInt, Tuple, Union
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType
@property
def micro(self) -> int:
    """The third item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").micro
        3
        >>> Version("1").micro
        0
        """
    return self.release[2] if len(self.release) >= 3 else 0