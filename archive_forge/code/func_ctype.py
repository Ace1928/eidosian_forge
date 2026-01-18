from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
@property
def ctype(self):
    """Complex type."""
    return np.complex64 if self.use_csingle else np.complex128