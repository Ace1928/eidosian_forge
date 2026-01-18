from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union
import numpy as np
from cirq import ops, protocols, value
from cirq._compat import proper_repr
@property
def gate_type(self) -> Type['cirq.Gate']:
    return self._gate_type