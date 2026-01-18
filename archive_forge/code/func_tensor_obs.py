from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
@property
def tensor_obs(self):
    """Tensor product observable matching ``use_csingle`` precision."""
    if self._use_mpi:
        return self.tensor_prod_obs_mpi_c64 if self.use_csingle else self.tensor_prod_obs_mpi_c128
    return self.tensor_prod_obs_c64 if self.use_csingle else self.tensor_prod_obs_c128