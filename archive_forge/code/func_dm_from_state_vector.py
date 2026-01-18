import functools
import itertools
from string import ascii_letters as ABC
from autoray import numpy as np
from numpy import float64
import pennylane as qml
from . import single_dispatch  # pylint:disable=unused-import
from .matrix_manipulation import _permute_dense_matrix
from .multi_dispatch import diag, dot, scatter_element_add, einsum, get_interface
from .utils import is_abstract, allclose, cast, convert_like, cast_like
def dm_from_state_vector(state, check_state=False, c_dtype='complex128'):
    """
    Convenience function to compute a (full) density matrix from
    a state vector.

    Args:
        state (tensor_like): 1D or 2D tensor state vector. This tensor should of size ``(2**N,)``
            or ``(batch_dim, 2**N)``, for some integer value ``N``.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Density matrix of size ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``

    **Example**

    >>> x = np.array([1, 0, 1j, 0]) / np.sqrt(2)
    >>> dm_from_state_vector(x)
    array([[0.5+0.j , 0. +0.j , 0. -0.5j, 0. +0.j ],
           [0. +0.j , 0. +0.j , 0. +0.j , 0. +0.j ],
           [0. +0.5j, 0. +0.j , 0.5+0.j , 0. +0.j ],
           [0. +0.j , 0. +0.j , 0. +0.j , 0. +0.j ]])

    """
    num_wires = int(np.log2(np.shape(state)[-1]))
    return reduce_statevector(state, indices=list(range(num_wires)), check_state=check_state, c_dtype=c_dtype)