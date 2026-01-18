from __future__ import annotations
from collections import defaultdict
from functools import partial
from itertools import chain, repeat
from typing import Callable, Iterable, Literal, Mapping
import numpy as np
from numpy.typing import NDArray
from qiskit.result import Counts
from .shape import ShapedMixin, ShapeInput, shape_tuple
@property
def num_bits(self) -> int:
    """The number of bits in the register that this array stores data for.

        For example, a ``ClassicalRegister(5, "meas")`` would result in ``num_bits=5``.
        """
    return self._num_bits