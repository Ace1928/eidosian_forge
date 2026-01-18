import functools
from collections.abc import Sequence
import autoray as ar
import numpy as onp
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from numpy import ndarray
from . import single_dispatch  # pylint:disable=unused-import
from .utils import cast, cast_like, get_interface, requires_grad
@multi_dispatch(argnum=[0, 1])
def frobenius_inner_product(A, B, normalize=False, like=None):
    """Frobenius inner product between two matrices.

    .. math::

        \\langle A, B \\rangle_F = \\sum_{i,j=1}^n A_{ij} B_{ij} = \\operatorname{tr} (A^T B)

    The Frobenius inner product is equivalent to the Hilbert-Schmidt inner product for
    matrices with real-valued entries.

    Args:
        A (tensor_like[float]): First matrix, assumed to be a square array.
        B (tensor_like[float]): Second matrix, assumed to be a square array.
        normalize (bool): If True, divide the inner product by the Frobenius norms of A and B.

    Returns:
        float: Frobenius inner product of A and B

    **Example**

    >>> A = np.random.random((3,3))
    >>> B = np.random.random((3,3))
    >>> qml.math.frobenius_inner_product(A, B)
    3.091948202943376
    """
    A, B = np.coerce([A, B], like=like)
    inner_product = np.sum(A * B)
    if normalize:
        norm = np.sqrt(np.sum(A * A) * np.sum(B * B))
        inner_product = inner_product / norm
    return inner_product