from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def apply_z(self, axis: int, exponent: float=1, global_shift: float=0):
    if exponent % 2 != 0:
        if exponent % 0.5 != 0.0:
            raise ValueError('Z exponent must be half integer')
        effective_exponent = exponent % 2
        for _ in range(int(effective_exponent * 2)):
            self.M[axis, :] ^= self.G[axis, :]
            self.gamma[axis] = (self.gamma[axis] - 1) % 4
    self.omega *= _phase(exponent, global_shift)