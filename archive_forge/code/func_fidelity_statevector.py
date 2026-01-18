from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
def fidelity_statevector(state0, state1, check_state=False, c_dtype='complex128'):
    """Compute the fidelity for two states (given as state vectors) acting on quantum
    systems with the same size.

    The fidelity for two pure states given by state vectors :math:`\\ket{\\psi}` and :math:`\\ket{\\phi}`
    is defined as

    .. math::
        F( \\ket{\\psi} , \\ket{\\phi}) = \\left|\\braket{\\psi, \\phi}\\right|^2

    This is faster than calling :func:`pennylane.math.fidelity` on the density matrix
    representation of pure states.

    .. note::
        It supports all interfaces (NumPy, Autograd, Torch, TensorFlow and Jax). The second state is coerced
        to the type and dtype of the first state. The fidelity is returned in the type of the interface of the
        first state.

    Args:
        state0 (tensor_like): ``(2**N)`` or ``(batch_dim, 2**N)`` state vector.
        state1 (tensor_like): ``(2**N)`` or ``(batch_dim, 2**N)`` state vector.
        check_state (bool): If True, the function will check the validity of both states; that is,
            the shape and the norm
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Fidelity between the two quantum states.

    **Example**

    Two state vectors can be used as arguments and the fidelity (overlap) is returned, e.g.:

    >>> state0 = [0.98753537-0.14925137j, 0.00746879-0.04941796j]
    >>> state1 = [0.99500417+0.j, 0.09983342+0.j]
    >>> qml.math.fidelity_statevector(state0, state1)
    0.9905158135644924

    .. seealso:: :func:`pennylane.math.fidelity` and :func:`pennylane.qinfo.transforms.fidelity`

    """
    state0 = cast(state0, dtype=c_dtype)
    state1 = cast(state1, dtype=c_dtype)
    if check_state:
        _check_state_vector(state0)
        _check_state_vector(state1)
    if qml.math.shape(state0)[-1] != qml.math.shape(state1)[-1]:
        raise qml.QuantumFunctionError('The two states must have the same number of wires.')
    batched0 = len(qml.math.shape(state0)) > 1
    batched1 = len(qml.math.shape(state1)) > 1
    indices0 = 'ab' if batched0 else 'b'
    indices1 = 'ab' if batched1 else 'b'
    target = 'a' if batched0 or batched1 else ''
    overlap = qml.math.einsum(f'{indices0},{indices1}->{target}', state0, qml.math.conj(state1), optimize='greedy')
    overlap = qml.math.abs(overlap) ** 2
    return overlap