import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import QubitUnitary
def probs_to_unitary(probs):
    """Calculates the unitary matrix corresponding to an input probability distribution.

    For a given distribution :math:`p(i)`, this function returns the unitary :math:`\\mathcal{A}`
    that transforms the :math:`|0\\rangle` state as

    .. math::

        \\mathcal{A} |0\\rangle = \\sum_{i} \\sqrt{p(i)} |i\\rangle,

    so that measuring the resulting state in the computational basis will give the state
    :math:`|i\\rangle` with probability :math:`p(i)`. Note that the returned unitary matrix is
    real and hence an orthogonal matrix.

    Args:
        probs (array): input probability distribution as a flat array

    Returns:
        array: unitary

    Raises:
        ValueError: if the input array is not flat or does not correspond to a probability
            distribution

    **Example:**

    >>> p = np.ones(4) / 4
    >>> probs_to_unitary(p)
    array([[ 0.5       ,  0.5       ,  0.5       ,  0.5       ],
           [ 0.5       , -0.83333333,  0.16666667,  0.16666667],
           [ 0.5       ,  0.16666667, -0.83333333,  0.16666667],
           [ 0.5       ,  0.16666667,  0.16666667, -0.83333333]])
    """
    if not qml.math.is_abstract(sum(probs)):
        if not qml.math.allclose(sum(probs), 1) or min(probs) < 0:
            raise ValueError('A valid probability distribution of non-negative numbers that sum to one must be input')
    psi = qml.math.sqrt(probs)
    overlap = psi[0]
    denominator = qml.math.sqrt(2 + 2 * overlap)
    psi = qml.math.set_index(psi, 0, psi[0] + 1)
    psi /= denominator
    dim = len(probs)
    return 2 * qml.math.outer(psi, psi) - np.eye(dim)