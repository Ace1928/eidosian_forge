from typing import Optional, TYPE_CHECKING
import numpy as np
from cirq import linalg, value
def random_superposition(dim: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> np.ndarray:
    """Returns a random unit-length vector from the uniform distribution.

    Args:
        dim: The dimension of the vector.
        random_state: A seed (int) or `np.random.RandomState` class to use when
            generating random values. If not set, defaults to using the module
            methods in `np.random`.

    Returns:
        The sampled unit-length vector.
    """
    random_state = value.parse_random_state(random_state)
    state_vector = random_state.randn(dim).astype(complex)
    state_vector += 1j * random_state.randn(dim)
    state_vector /= np.linalg.norm(state_vector)
    return state_vector