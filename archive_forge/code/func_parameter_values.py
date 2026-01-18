from __future__ import annotations
from collections.abc import Mapping
from typing import Tuple, Union
from numbers import Integral
from qiskit import QuantumCircuit
from .bindings_array import BindingsArray, BindingsArrayLike
from .shape import ShapedMixin
@property
def parameter_values(self) -> BindingsArray:
    """A bindings array."""
    return self._parameter_values