from functools import lru_cache, reduce
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops.qubit.parametric_ops_multi_qubit import PauliRot
@lru_cache
def pauli_basis_matrices(num_wires):
    """Compute all elements of the Pauli basis of the Lie algebra :math:`\\mathfrak{su}(N)`
    as a single, dense tensor.

    Args:
        num_wires (int): The number of wires on which the associated Pauli group acts.

    Returns:
        ndarray: All Pauli basis elements of :math:`\\mathfrak{su}(N)`.

    The basis has :math:`d=4^n-1` elements for :math:`n` qubits, yielding an output tensor
    with shape ``(4**num_wires-1, 2**num_wires, 2**num_wires)`` and :math:`16^n-4^n` entries.
    The identity Pauli word :math:`I^{\\otimes n}` does not belong to :math:`\\mathfrak{su}(N)`
    and therefore is not included.

    The basis elements are ordered (choose the description that suits you most)

      - lexicographically.

      - such that the term acting on the last qubit changes fastest, the one acting on the first
        qubit changes slowest when iterating through the output.

      - such that the basis index, written in base :math:`4`, contains the indices for the list
        ``["I", "X", "Y", "Z"]``, in the order of the qubits

      - such that for three qubits, the first basis elements are the Pauli words
        ``"IIX", ""IIY", "IIZ", "IXI", "IXX", "IXY", "IXZ", "IYI"...``

    .. note::

        Note that this method internally handles a complex-valued tensor of size
        ``(4**num_wires, 2**num_wires, 2**num_wires)``, which requires at least
        ``4 ** (2 * num_wires - 13)`` GB of memory (at default precision).

    **Example**

    >>> pauli_basis_matrices(1)
    array([[[ 0.+0.j,  1.+0.j],
            [ 1.+0.j,  0.+0.j]],
           [[ 0.+0.j, -0.-1.j],
            [ 0.+1.j,  0.+0.j]],
           [[ 1.+0.j,  0.+0.j],
            [ 0.+0.j, -1.+0.j]]])

    >>> pauli_basis_matrices(3).shape
    (63, 8, 8)
    """
    if num_wires < 1:
        raise ValueError(f'Require at least one wire, got {num_wires}.')
    if num_wires > 7:
        raise ValueError(f'Creating the Pauli basis tensor for more than 7 wires (got {num_wires}) is deactivated.')
    return reduce(np.kron, (_pauli_matrices for _ in range(num_wires)))[1:]