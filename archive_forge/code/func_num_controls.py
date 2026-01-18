from typing import (
import numpy as np
from cirq import protocols, value, _import
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def num_controls(self) -> int:
    return len(self.control_qid_shape)