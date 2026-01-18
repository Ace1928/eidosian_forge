from typing import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size
def recycle(self) -> None:
    if self._state < 2:
        for _ in self:
            pass
    if self._state == 2:
        self._state = 1