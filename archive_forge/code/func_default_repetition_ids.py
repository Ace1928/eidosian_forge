import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
def default_repetition_ids(repetitions: IntParam) -> Optional[List[str]]:
    if isinstance(repetitions, INT_CLASSES) and abs(repetitions) != 1:
        abs_repetitions: int = abs(int(repetitions))
        return [str(i) for i in range(abs_repetitions)]
    return None