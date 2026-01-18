from __future__ import annotations
import contextlib
from abc import ABC
from typing import (
def hincr(self, key: str, field: str, amount: AmountT=1) -> AmountT:
    """
        Increments the given key
        """
    func = self.kdb.hincrby if isinstance(amount, int) else self.kdb.hincrbyfloat
    return func(self.get_key(key), field, amount=amount)