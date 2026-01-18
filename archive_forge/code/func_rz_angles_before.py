import cmath
import math
from typing import AbstractSet, Any, Dict, Optional, Tuple
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import gate_features, raw_types
@property
def rz_angles_before(self) -> Tuple['cirq.TParamVal', 'cirq.TParamVal']:
    """Returns 2-tuple of phase angles applied to qubits before FSimGate."""
    b0 = (-self.gamma + self.zeta + self.chi) / 2.0
    b1 = (-self.gamma - self.zeta - self.chi) / 2.0
    return (b0, b1)