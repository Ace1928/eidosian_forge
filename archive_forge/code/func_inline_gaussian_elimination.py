import abc
import numbers
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, linalg, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, identity, pauli_gates, global_phase_op, pauli_string
from cirq.type_workarounds import NotImplementedType
@classmethod
def inline_gaussian_elimination(cls, rows: 'List[MutableDensePauliString]') -> None:
    if not rows:
        return
    height = len(rows)
    width = len(rows[0])
    next_row = 0
    for col in range(width):
        for held in [DensePauliString.Z_VAL, DensePauliString.X_VAL]:
            for k in range(next_row, height):
                if (rows[k].pauli_mask[col] or held) != held:
                    pivot_row = k
                    break
            else:
                continue
            for k in range(height):
                if k != pivot_row:
                    if (rows[k].pauli_mask[col] or held) != held:
                        rows[k].__imul__(rows[pivot_row])
            if pivot_row != next_row:
                rows[next_row], rows[pivot_row] = (rows[pivot_row], rows[next_row])
            next_row += 1