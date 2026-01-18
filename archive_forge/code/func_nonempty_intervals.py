from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import devices, ops, protocols
def nonempty_intervals(intervals: Sequence[Tuple[float, float]]) -> Sequence[Tuple[float, float]]:
    return tuple(((a, b) for a, b in intervals if a < b))