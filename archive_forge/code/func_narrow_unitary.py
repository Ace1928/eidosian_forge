import dataclasses
import cirq
import numpy as np
from cirq import ops, qis, protocols
def narrow_unitary(self) -> np.ndarray:
    """Narrowed unitary corresponding to the unitary effect applied on target qubits."""
    return _matrix_for_phasing_state(self.target_bitsize, self.phase_state, -1)