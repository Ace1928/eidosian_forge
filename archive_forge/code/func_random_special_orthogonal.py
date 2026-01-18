from typing import Optional, TYPE_CHECKING
import numpy as np
from cirq import linalg, value
def random_special_orthogonal(dim: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> np.ndarray:
    """Returns a random special orthogonal matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        random_state: A seed (int) or `np.random.RandomState` class to use when
            generating random values. If not set, defaults to using the module
            methods in `np.random`.

    Returns:
        The sampled special orthogonal matrix.
    """
    m = random_orthogonal(dim, random_state=random_state)
    if np.linalg.det(m) < 0:
        m[0, :] *= -1
    return m