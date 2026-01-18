from typing import cast, List, Optional, Sequence, Union, Tuple
import numpy as np
from cirq.linalg import tolerance, transformations
from cirq import value
def is_cptp(*, kraus_ops: Sequence[np.ndarray], rtol: float=1e-05, atol: float=1e-08):
    """Determines if a channel is completely positive trace preserving (CPTP).

    A channel composed of Kraus operators K[0:n] is a CPTP map if the sum of
    the products `adjoint(K[i]) * K[i])` is equal to 1.

    Args:
        kraus_ops: The Kraus operators of the channel to check.
        rtol: The relative tolerance on equality.
        atol: The absolute tolerance on equality.
    """
    sum_ndarray = cast(np.ndarray, sum((matrix.T.conj() @ matrix for matrix in kraus_ops)))
    return np.allclose(sum_ndarray, np.eye(sum_ndarray.shape[0], sum_ndarray.shape[1]), rtol=rtol, atol=atol)