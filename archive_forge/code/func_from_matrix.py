from typing import AbstractSet, Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numbers
import numpy as np
import sympy
from cirq import value, ops, protocols, linalg
from cirq.ops import raw_types
from cirq._compat import proper_repr
@staticmethod
def from_matrix(mat: np.ndarray) -> 'cirq.PhasedXZGate':
    pre_phase, rotation, post_phase = linalg.deconstruct_single_qubit_matrix_into_angles(mat)
    pre_phase /= np.pi
    post_phase /= np.pi
    rotation /= np.pi
    pre_phase -= 0.5
    post_phase += 0.5
    return PhasedXZGate(x_exponent=rotation, axis_phase_exponent=-pre_phase, z_exponent=post_phase + pre_phase)._canonical()