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
def min_entropy(state, indices, base=None, check_state=False, c_dtype='complex128'):
    """Compute the minimum entropy from a density matrix.

    .. math::
        S_{\\text{min}}( \\rho ) = -\\log( \\max_{i} ( p_{i} ))

    Args:
        state (tensor_like): Density matrix of shape ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``.
        indices (list(int)): List of indices in the considered subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: The minimum entropy of the considered subsystem.

    **Example**

    The minimum entropy of a subsystem for any state vector can be obtained by first calling
    :func:`~.math.dm_from_state_vector` on the input. Here is an example for the
    maximally entangled state, where the subsystem entropy is maximal (default base for log is exponential).

    >>> x = [1, 0, 0, 1] / np.sqrt(2)
    >>> x = dm_from_state_vector(x)
    >>> min_entropy(x, indices=[0])
    0.6931472

    The logarithm base can be changed. For example:

    >>> min_entropy(x, indices=[0], base=2)
    1.0

    The minimum entropy can be obtained by providing a quantum state as a density matrix. For example:

    >>> y = [[1/2, 0, 0, 1/2], [0, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]]
    >>> min_entropy(y, indices=[0])
    0.6931472

    The Von Neumann entropy is always greater than the minimum entropy.

    >>> x = [np.cos(np.pi/8), 0, 0, -1j*np.sin(np.pi/8)]
    >>> x = dm_from_state_vector(x)
    >>> vn_entropy(x, indices=[1])
    0.4164955
    >>> min_entropy(x, indices=[1])
    0.1583472

    """
    density_matrix = reduce_dm(state, indices, check_state, c_dtype)
    minimum_entropy = _compute_min_entropy(density_matrix, base)
    return minimum_entropy