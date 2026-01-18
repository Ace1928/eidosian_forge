import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
@property
def qubit_map(self) -> Mapping['cirq.Qid', 'cirq.Qid']:
    return self._qubit_map