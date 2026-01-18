import functools
from typing import Union, TYPE_CHECKING
import numpy as np
from cirq._doc import document
def kron_with_controls(*factors: Union[np.ndarray, complex, float]) -> np.ndarray:
    """Computes the kronecker product of a sequence of values and control tags.

    Use `cirq.CONTROL_TAG` to represent controls. Any entry of the output
    corresponding to a situation where the control is not satisfied will
    be overwritten by identity matrix elements.

    The control logic works by imbuing NaN with the meaning "failed to meet one
    or more controls". The normal kronecker product then spreads the per-item
    NaNs to all the entries in the product that need to be replaced by identity
    matrix elements. This method rewrites those NaNs. Thus CONTROL_TAG can be
    the matrix [[NaN, 0], [0, 1]] or equivalently [[NaN, NaN], [NaN, 1]].

    Because this method re-interprets NaNs as control-failed elements, it won't
    propagate error-indicating NaNs from its input to its output in the way
    you'd otherwise expect.

    Examples:

        ```
        result = cirq.kron_with_controls(
            cirq.CONTROL_TAG,
            cirq.unitary(cirq.X))
        print(result.astype(np.int32))

        # prints:
        # [[1 0 0 0]
        #  [0 1 0 0]
        #  [0 0 0 1]
        #  [0 0 1 0]]
        ```

    Args:
        *factors: The matrices, tensors, scalars, and/or control tags to combine
            together using np.kron.

    Returns:
        The resulting matrix.
    """
    product = kron(*factors)
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            if np.isnan(product[i, j]):
                product[i, j] = 1 if i == j else 0
    return product