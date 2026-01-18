from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
@property
def sv_type(self):
    """State vector matching ``use_csingle`` precision (and MPI if it is supported)."""
    if self._use_mpi:
        return self.statevector_mpi_c64 if self.use_csingle else self.statevector_mpi_c128
    return self.statevector_c64 if self.use_csingle else self.statevector_c128