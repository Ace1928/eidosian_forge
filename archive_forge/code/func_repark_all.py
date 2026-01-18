from __future__ import annotations
import math
from collections import OrderedDict
from typing import TYPE_CHECKING
import attrs
from .. import _core
from .._util import final
def repark_all(self, new_lot: ParkingLot) -> None:
    """Move all parked tasks from one :class:`ParkingLot` object to
        another.

        See :meth:`repark` for details.

        """
    return self.repark(new_lot, count=len(self))