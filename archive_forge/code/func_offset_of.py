import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
def offset_of(self, other, n):
    prior = (self.low, self.high)
    self.low = min(self.low, other.low + n)
    self.high = max(self.high, other.high + n)
    if (self.low, self.high) != prior:
        self.fixed_point.value = False