from typing import AbstractSet, Any, Dict, Tuple, Optional, Sequence, TYPE_CHECKING
import numpy as np
import sympy
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, common_gates
@property
def diag_angles_radians(self) -> Tuple[value.TParamVal, ...]:
    return self._diag_angles_radians