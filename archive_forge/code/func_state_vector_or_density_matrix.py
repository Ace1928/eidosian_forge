from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def state_vector_or_density_matrix(self) -> np.ndarray:
    """Return the state vector or density matrix of this state.

        If the state is a denity matrix, return the density matrix. Otherwise, return the state
        vector.
        """
    state_vector = self.state_vector()
    if state_vector is not None:
        return state_vector
    return self.data