from functools import reduce, singledispatch
from itertools import product
from operator import matmul
from typing import Union, Tuple
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum
from pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform
from .pauli_arithmetic import I, PauliSentence, PauliWord, X, Y, Z, op_map
from .utils import is_pauli_word
def pauli_decompose(H, hide_identity=False, wire_order=None, pauli=False, check_hermitian=True) -> Union[Hamiltonian, PauliSentence]:
    """Decomposes a Hermitian matrix into a linear combination of Pauli operators.

    Args:
        H (tensor_like[complex]): a Hermitian matrix of dimension :math:`2^n\\times 2^n`.
        hide_identity (bool): does not include the Identity observable within
            the tensor products of the decomposition if ``True``.
        wire_order (list[Union[int, str]]): the ordered list of wires with respect
            to which the operator is represented as a matrix.
        pauli (bool): return a :class:`~.PauliSentence` instance if ``True``.
        check_hermitian (bool): check if the provided matrix is Hermitian if ``True``.

    Returns:
        Union[~.Hamiltonian, ~.PauliSentence]: the matrix decomposed as a linear combination
        of Pauli operators, returned either as a :class:`~.Hamiltonian` or :class:`~.PauliSentence`
        instance.

    **Example:**

    We can use this function to compute the Pauli operator decomposition of an arbitrary Hermitian
    matrix:

    >>> A = np.array(
    ... [[-2, -2+1j, -2, -2], [-2-1j,  0,  0, -1], [-2,  0, -2, -1], [-2, -1, -1,  0]])
    >>> H = qml.pauli_decompose(A)
    >>> print(H)
    (-1.5) [I0 X1]
    + (-1.5) [X0 I1]
    + (-1.0) [I0 I1]
    + (-1.0) [I0 Z1]
    + (-1.0) [X0 X1]
    + (-0.5) [I0 Y1]
    + (-0.5) [X0 Z1]
    + (-0.5) [Z0 X1]
    + (-0.5) [Z0 Y1]
    + (1.0) [Y0 Y1]

    We can return a :class:`~.PauliSentence` instance by using the keyword argument ``pauli=True``:

    >>> ps = qml.pauli_decompose(A, pauli=True)
    >>> print(ps)
    -1.0 * I
    + -1.5 * X(1)
    + -0.5 * Y(1)
    + -1.0 * Z(1)
    + -1.5 * X(0)
    + -1.0 * X(0) @ X(1)
    + -0.5 * X(0) @ Z(1)
    + 1.0 * Y(0) @ Y(1)
    + -0.5 * Z(0) @ X(1)
    + -0.5 * Z(0) @ Y(1)

    By default the wires are numbered [0, 1, ..., n], but we can also set custom wires using the
    ``wire_order`` argument:

    >>> ps = qml.pauli_decompose(A, pauli=True, wire_order=['a', 'b'])
    >>> print(ps)
    -1.0 * I
    + -1.5 * X(b)
    + -0.5 * Y(b)
    + -1.0 * Z(b)
    + -1.5 * X(a)
    + -1.0 * X(a) @ X(b)
    + -0.5 * X(a) @ Z(b)
    + 1.0 * Y(a) @ Y(b)
    + -0.5 * Z(a) @ X(b)
    + -0.5 * Z(a) @ Y(b)

    .. details::
        :title: Theory
        :href: theory

        This method internally uses a generalized decomposition routine to convert the matrix to a
        weighted sum of Pauli words acting on :math:`n` qubits in time :math:`O(n 4^n)`. The input
        matrix is written as a quantum state in the computational basis following the
        `channel-state duality <https://en.wikipedia.org/wiki/Channel-state_duality>`_.
        A Bell basis transformation is then performed using the
        `Walsh-Hadamard transform <https://en.wikipedia.org/wiki/Hadamard_transform>`_, after which
        coefficients for each of the :math:`4^n` Pauli words are computed while accounting for the
        phase from each ``PauliY`` term occurring in the word.

    """
    shape = qml.math.shape(H)
    n = int(qml.math.log2(shape[0]))
    N = 2 ** n
    if check_hermitian:
        if shape != (N, N):
            raise ValueError('The matrix should have shape (2**n, 2**n), for any qubit number n>=1')
        if not qml.math.allclose(H, qml.math.conj(qml.math.transpose(H))):
            raise ValueError('The matrix is not Hermitian')
    coeffs, obs = _generalized_pauli_decompose(H, hide_identity=hide_identity, wire_order=wire_order, pauli=pauli, padding=True)
    if check_hermitian:
        coeffs = qml.math.real(coeffs)
    if pauli:
        return PauliSentence({PauliWord({w: o for o, w in obs_n_wires}): coeff for coeff, obs_n_wires in zip(coeffs, obs)})
    return Hamiltonian(coeffs, obs)