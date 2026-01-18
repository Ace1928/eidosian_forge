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
def num_shots(self) -> int:
    """The number of shots sampled from the register in each configuration.

        More precisely, the length of the second last axis of :attr:`~.array`.
        """
    return self._array.shape[-2]