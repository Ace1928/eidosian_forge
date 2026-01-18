from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def state_tensor(self) -> Optional[np.ndarray]:
    """Return the state tensor of this state.

        A state tensor stores the amplitudes of a pure state as an array with
        shape equal to the qid shape of the state.
        If the state is a density matrix, this method returns None.
        """
    if self._is_density_matrix():
        return None
    return np.reshape(self.data, self.qid_shape)