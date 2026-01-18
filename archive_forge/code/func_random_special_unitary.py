from typing import Optional, TYPE_CHECKING
import numpy as np
from cirq import linalg, value
def random_special_unitary(dim: int, *, random_state: Optional[np.random.RandomState]=None) -> np.ndarray:
    """Returns a random special unitary distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        random_state: A seed (int) or `np.random.RandomState` class to use when
            generating random values. If not set, defaults to using the module
            methods in `np.random`.

    Returns:
        The sampled special unitary.
    """
    r = random_unitary(dim, random_state=random_state)
    r[0, :] /= np.linalg.det(r)
    return r