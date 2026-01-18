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
def max_entropy(state, indices, base=None, check_state=False, c_dtype='complex128'):
    """Compute the maximum entropy of a density matrix on a given subsystem. It supports all
    interfaces (NumPy, Autograd, Torch, TensorFlow and Jax).

    .. math::
        S_{\\text{max}}( \\rho ) = \\log( \\text{rank} ( \\rho ))

    Args:
        state (tensor_like): Density matrix of shape ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``.
        indices (list(int)): List of indices in the considered subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: The maximum entropy of the considered subsystem.

    **Example**

    The maximum entropy of a subsystem for any state vector can be obtained by first calling
    :func:`~.math.dm_from_state_vector` on the input. Here is an example for the
    maximally entangled state, where the subsystem entropy is maximal (default base for log is exponential).

    >>> x = [1, 0, 0, 1] / np.sqrt(2)
    >>> x = dm_from_state_vector(x)
    >>> max_entropy(x, indices=[0])
    0.6931472

    The logarithm base can be changed. For example:

    >>> max_entropy(x, indices=[0], base=2)
    1.0

    The maximum entropy can be obtained by providing a quantum state as a density matrix. For example:

    >>> y = [[1/2, 0, 0, 1/2], [0, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]]
    >>> max_entropy(y, indices=[0])
    0.6931472

    The maximum entropy is always greater or equal to the Von Neumann entropy. In this maximally
    entangled example, they are equal:

    >>> vn_entropy(x, indices=[0])
    0.6931472

    However, in general, the Von Neumann entropy is lower:

    >>> x = [np.cos(np.pi/8), 0, 0, -1j*np.sin(np.pi/8)]
    >>> x = dm_from_state_vector(x)
    >>> vn_entropy(x, indices=[1])
    0.4164955
    >>> max_entropy(x, indices=[1])
    0.6931472

    """
    density_matrix = reduce_dm(state, indices, check_state, c_dtype)
    maximum_entropy = _compute_max_entropy(density_matrix, base)
    return maximum_entropy