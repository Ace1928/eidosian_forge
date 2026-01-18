import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import QubitUnitary
def func_to_unitary(func, M):
    """Calculates the unitary that encodes a function onto an ancilla qubit register.

    Consider a function defined on the set of integers :math:`X = \\{0, 1, \\ldots, M - 1\\}` whose
    output is bounded in the interval :math:`[0, 1]`, i.e., :math:`f: X \\rightarrow [0, 1]`.

    The ``func_to_unitary`` function returns a unitary :math:`\\mathcal{R}` that performs the
    transformation:

    .. math::

        \\mathcal{R} |i\\rangle \\otimes |0\\rangle = |i\\rangle\\otimes \\left(\\sqrt{1 - f(i)}|0\\rangle +
        \\sqrt{f(i)} |1\\rangle\\right).

    In other words, for a given input state :math:`|i\\rangle \\otimes |0\\rangle`, this unitary
    encodes the amplitude :math:`\\sqrt{f(i)}` onto the :math:`|1\\rangle` state of the ancilla qubit.
    Hence, measuring the ancilla qubit will result in the :math:`|1\\rangle` state with probability
    :math:`f(i)`.

    Args:
        func (callable): a function defined on the set of integers
            :math:`X = \\{0, 1, \\ldots, M - 1\\}` with output value inside :math:`[0, 1]`
        M (int): the number of integers that the function is defined on

    Returns:
        array: the :math:`\\mathcal{R}` unitary

    Raises:
        ValueError: if func is not bounded with :math:`[0, 1]` for all :math:`X`

    **Example:**

    >>> func = lambda i: np.sin(i) ** 2
    >>> M = 16
    >>> func_to_unitary(func, M)
    array([[ 1.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        , -1.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.54030231, ...,  0.        ,
             0.        ,  0.        ],
           ...,
           [ 0.        ,  0.        ,  0.        , ..., -0.13673722,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.75968791,  0.65028784],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.65028784, -0.75968791]])
    """
    unitary = np.zeros((2 * M, 2 * M))
    fs = [func(i) for i in range(M)]
    if not qml.math.is_abstract(fs[0]):
        if min(fs) < 0 or max(fs) > 1:
            raise ValueError('func must be bounded within the interval [0, 1] for the range of input values')
    for i, f in enumerate(fs):
        unitary = qml.math.set_index(unitary, (2 * i, 2 * i), qml.math.sqrt(1 - f))
        unitary = qml.math.set_index(unitary, (2 * i + 1, 2 * i), qml.math.sqrt(f))
        unitary = qml.math.set_index(unitary, (2 * i, 2 * i + 1), qml.math.sqrt(f))
        unitary = qml.math.set_index(unitary, (2 * i + 1, 2 * i + 1), -qml.math.sqrt(1 - f))
    return unitary