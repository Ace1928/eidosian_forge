from __future__ import annotations
import copy
from abc import ABC
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from .mixins import GroupMixin
@property
def qargs(self):
    """Return the qargs for the operator."""
    return self._qargs