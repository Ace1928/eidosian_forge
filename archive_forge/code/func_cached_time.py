from __future__ import annotations
import abc
import pickle
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union
from ..util.typing import Self
@property
def cached_time(self) -> float:
    """The epoch (floating point time value) stored when this payload was
        cached.

        .. versionadded:: 1.3

        """
    return cast(float, self.metadata['ct'])