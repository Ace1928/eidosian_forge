import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
def base_operation(self) -> 'cirq.CircuitOperation':
    """Returns a copy of this operation with only the wrapped circuit.

        Key and qubit mappings, parameter values, and repetitions are not copied.
        """
    return CircuitOperation(self.circuit)