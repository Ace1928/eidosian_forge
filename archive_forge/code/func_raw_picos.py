from datetime import timedelta
from typing import Union, overload
from cirq.value.duration import Duration
def raw_picos(self) -> float:
    """The timestamp's location in picoseconds from arbitrary time zero."""
    return self._picos