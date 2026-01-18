from __future__ import annotations
import copy
from abc import ABC
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from .mixins import GroupMixin
def input_dims(self, qargs=None):
    """Return tuple of input dimension for specified subsystems."""
    return self._op_shape.dims_r(qargs)