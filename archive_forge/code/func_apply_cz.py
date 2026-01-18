from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def apply_cz(self, control_axis: int, target_axis: int, exponent: float=1, global_shift: float=0):
    if exponent % 2 != 0:
        if exponent % 1 != 0:
            raise ValueError('CZ exponent must be integer')
        self.M[control_axis, :] ^= self.G[target_axis, :]
        self.M[target_axis, :] ^= self.G[control_axis, :]
    self.omega *= _phase(exponent, global_shift)