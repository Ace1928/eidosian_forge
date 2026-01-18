from typing import Optional, List, Tuple, Iterable, Callable, Union, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
from .base_readout_mitigator import BaseReadoutMitigator
from .utils import counts_probability_vector, z_diagonal, str2diag
def stddev_upper_bound(self, shots: int, qubits: List[int]=None):
    """Return an upper bound on standard deviation of expval estimator.

        Args:
            shots: Number of shots used for expectation value measurement.
            qubits: qubits being measured for operator expval.

        Returns:
            float: the standard deviation upper bound.
        """
    gamma = self._compute_gamma(qubits=qubits)
    return gamma / np.sqrt(shots)