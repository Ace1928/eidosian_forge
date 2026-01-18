from __future__ import annotations
from typing import (
from pandas.core.interchange.dataframe_protocol import (
@property
def ptr(self) -> int:
    """
        Pointer to start of the buffer as an integer.
        """
    return self._x.__array_interface__['data'][0]