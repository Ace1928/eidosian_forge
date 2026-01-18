from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, protocols, study
from cirq.experiments.qubit_characterizations import TomographyResult
def fit_density_matrix(self, counts: np.ndarray) -> TomographyResult:
    """Solves equation mat * rho = probs.

        Args:
            counts: A 2D array where each row contains measured counts
                of all n-qubit bitstrings for the corresponding pre-rotations
                in `rot_sweep`.  The order of the probabilities corresponds to
                to `rot_sweep` and the order of the bit strings corresponds to
                increasing integers up to 2**(num_qubits)-1.

        Returns:
            `TomographyResult` with density matrix corresponding to solution of
            this system.
        """
    probs = counts / np.sum(counts, axis=1)[:, np.newaxis]
    c, _, _, _ = np.linalg.lstsq(self.mat, np.asarray(probs).flat, rcond=-1)
    rho = c.reshape((2 ** self.num_qubits, 2 ** self.num_qubits))
    return TomographyResult(rho)