from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def eye_tensor(half_shape: Tuple[int, ...], *, dtype: 'DTypeLike') -> np.ndarray:
    """Returns an identity matrix reshaped into a tensor.

    Args:
        half_shape: A tuple representing the number of quantum levels of each
            qubit the returned matrix applies to.  `half_shape` is (2, 2, 2) for
            a three-qubit identity operation tensor.
        dtype: The numpy dtype of the new array.

    Returns:
        The created numpy array with shape `half_shape + half_shape`.
    """
    identity = np.eye(np.prod(half_shape, dtype=np.int64).item(), dtype=dtype)
    identity.shape = half_shape * 2
    return identity