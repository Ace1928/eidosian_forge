from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
@property
def sparse_hamiltonian_obs(self):
    """SparseHamiltonian observable matching ``use_csingle`` precision."""
    if self._use_mpi:
        return self.sparse_hamiltonian_mpi_c64 if self.use_csingle else self.sparse_hamiltonian_mpi_c128
    return self.sparse_hamiltonian_c64 if self.use_csingle else self.sparse_hamiltonian_c128