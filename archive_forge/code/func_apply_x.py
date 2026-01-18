from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def apply_x(self, axis: int, exponent: float=1, global_shift: float=0):
    if exponent % 2 != 0:
        if exponent % 0.5 != 0.0:
            raise ValueError('X exponent must be half integer')
        self.apply_h(axis)
        self.apply_z(axis, exponent)
        self.apply_h(axis)
    self.omega *= _phase(exponent, global_shift)