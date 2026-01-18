import itertools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix
from cirq import value
from cirq.ops import raw_types
def expectation_from_state_vector(self, state_vector: np.ndarray, qid_map: Mapping[raw_types.Qid, int]) -> complex:
    """Expectation of the projection from a state vector.

        Computes the expectation value of this ProjectorString on the provided state vector.

        Args:
            state_vector: An array representing a valid state vector.
            qid_map: A map from all qubits used in this ProjectorString to the
                indices of the qubits that `state_vector` is defined over.

        Returns:
            The expectation value of the input state.
        """
    _check_qids_dimension(qid_map.keys())
    num_qubits = len(qid_map)
    index = self._get_idx_to_keep(qid_map)
    return self._coefficient * np.sum(np.abs(np.reshape(state_vector, (2,) * num_qubits)[index]) ** 2)