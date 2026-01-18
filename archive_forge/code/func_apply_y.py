from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def apply_y(self, axis: int, exponent: float=1, global_shift: float=0):
    if exponent % 0.5 != 0.0:
        raise ValueError('Y exponent must be half integer')
    shift = _phase(exponent, global_shift)
    if exponent % 2 == 0:
        self.omega *= shift
    elif exponent % 2 == 0.5:
        self.apply_z(axis)
        self.apply_h(axis)
        self.omega *= shift * (1 + 1j) / 2 ** 0.5
    elif exponent % 2 == 1:
        self.apply_z(axis)
        self.apply_h(axis)
        self.apply_z(axis)
        self.apply_h(axis)
        self.omega *= shift * 1j
    elif exponent % 2 == 1.5:
        self.apply_h(axis)
        self.apply_z(axis)
        self.omega *= shift * (1 - 1j) / 2 ** 0.5