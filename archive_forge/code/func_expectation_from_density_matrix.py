import itertools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix
from cirq import value
from cirq.ops import raw_types
def expectation_from_density_matrix(self, state: np.ndarray, qid_map: Mapping[raw_types.Qid, int]) -> complex:
    """Expectation of the projection from a density matrix.

        Computes the expectation value of this ProjectorString on the provided state.

        Args:
            state: An array representing a valid  density matrix.
            qid_map: A map from all qubits used in this ProjectorString to the
                indices of the qubits that `state_vector` is defined over.

        Returns:
            The expectation value of the input state.
        """
    _check_qids_dimension(qid_map.keys())
    num_qubits = len(qid_map)
    index = self._get_idx_to_keep(qid_map) * 2
    result = np.reshape(state, (2,) * (2 * num_qubits))[index]
    while any(result.shape):
        result = np.trace(result, axis1=0, axis2=len(result.shape) // 2)
    return self._coefficient * result