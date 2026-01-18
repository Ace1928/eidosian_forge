from __future__ import annotations
from typing import Mapping, Union, Tuple
from collections.abc import Iterable, Mapping as _Mapping
from itertools import chain, islice
import numpy as np
from numpy.typing import ArrayLike
from qiskit.circuit import Parameter, QuantumCircuit
from .shape import ShapedMixin, ShapeInput, shape_tuple
def bind_all(self, circuit: QuantumCircuit) -> np.ndarray:
    """Return an object array of bound circuits with the same shape.

        Args:
            circuit: The circuit to bind.

        Returns:
            An object array of the same shape containing all bound circuits.
        """
    arr = np.empty(self.shape, dtype=object)
    for idx in np.ndindex(self.shape):
        arr[idx] = self.bind(circuit, idx)
    return arr