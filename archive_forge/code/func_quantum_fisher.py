from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
@partial(transform, is_informative=True)
def quantum_fisher(tape: qml.tape.QuantumTape, device, *args, **kwargs) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Returns a function that computes the quantum fisher information matrix (QFIM) of a given :class:`.QNode`.

    Given a parametrized quantum state :math:`|\\psi(\\bm{\\theta})\\rangle`, the quantum fisher information matrix (QFIM) quantifies how changes to the parameters :math:`\\bm{\\theta}`
    are reflected in the quantum state. The metric used to induce the QFIM is the fidelity :math:`f = |\\langle \\psi | \\psi' \\rangle|^2` between two (pure) quantum states.
    This leads to the following definition of the QFIM (see eq. (27) in `arxiv:2103.15191 <https://arxiv.org/abs/2103.15191>`_):

    .. math::

        \\text{QFIM}_{i, j} = 4 \\text{Re}\\left[ \\langle \\partial_i \\psi(\\bm{\\theta}) | \\partial_j \\psi(\\bm{\\theta}) \\rangle
        - \\langle \\partial_i \\psi(\\bm{\\theta}) | \\psi(\\bm{\\theta}) \\rangle \\langle \\psi(\\bm{\\theta}) | \\partial_j \\psi(\\bm{\\theta}) \\rangle \\right]

    with short notation :math:`| \\partial_j \\psi(\\bm{\\theta}) \\rangle := \\frac{\\partial}{\\partial \\theta_j}| \\psi(\\bm{\\theta}) \\rangle`.

    .. seealso::
        :func:`~.pennylane.metric_tensor`, :func:`~.pennylane.adjoint_metric_tensor`, :func:`~.pennylane.qinfo.transforms.classical_fisher`

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit that may have arbitrary return types.
        *args: In case finite shots are used, further arguments according to :func:`~.pennylane.metric_tensor` may be passed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the quantum Fisher information in the form of a tensor.

    .. note::

        ``quantum_fisher`` coincides with the ``metric_tensor`` with a prefactor of :math:`4`. Internally, :func:`~.pennylane.adjoint_metric_tensor` is used when executing on a device with
        exact expectations (``shots=None``) that inherits from ``"default.qubit"``. In all other cases, i.e. if a device with finite shots is used, the hardware compatible transform :func:`~.pennylane.metric_tensor` is used.
        Please refer to their respective documentations for details on the arguments.

    **Example**

    The quantum Fisher information matrix (QIFM) can be used to compute the `natural` gradient for `Quantum Natural Gradient Descent <https://arxiv.org/abs/1909.02108>`_.
    A typical scenario is optimizing the expectation value of a Hamiltonian:

    .. code-block:: python

        n_wires = 2

        dev = qml.device("default.qubit", wires=n_wires)

        H = 1.*qml.X(0) @ qml.X(1) - 0.5 * qml.Z(1)

        @qml.qnode(dev)
        def circ(params):
            qml.RY(params[0], wires=1)
            qml.CNOT(wires=(1,0))
            qml.RY(params[1], wires=1)
            qml.RZ(params[2], wires=1)
            return qml.expval(H)

        params = pnp.array([0.5, 1., 0.2], requires_grad=True)

    The natural gradient is then simply the QFIM multiplied by the gradient:

    >>> grad = qml.grad(circ)(params)
    >>> grad
    [ 0.59422561 -0.02615095 -0.05146226]
    >>> qfim = qml.qinfo.quantum_fisher(circ)(params)
    >>> qfim
    [[1.         0.         0.        ]
     [0.         1.         0.        ]
     [0.         0.         0.77517241]]
    >>> qfim @ grad
    tensor([ 0.59422561, -0.02615095, -0.03989212], requires_grad=True)

    When using real hardware or finite shots, ``quantum_fisher`` is internally calling :func:`~.pennylane.metric_tensor`.
    To obtain the full QFIM, we need an auxilary wire to perform the Hadamard test.

    >>> dev = qml.device("default.qubit", wires=n_wires+1, shots=1000)
    >>> @qml.qnode(dev)
    ... def circ(params):
    ...     qml.RY(params[0], wires=1)
    ...     qml.CNOT(wires=(1,0))
    ...     qml.RY(params[1], wires=1)
    ...     qml.RZ(params[2], wires=1)
    ...     return qml.expval(H)
    >>> qfim = qml.qinfo.quantum_fisher(circ)(params)

    Alternatively, we can fall back on the block-diagonal QFIM without the additional wire.

    >>> qfim = qml.qinfo.quantum_fisher(circ, approx="block-diag")(params)

    """
    if device.shots and isinstance(device, (DefaultQubitLegacy, DefaultQubit)):
        tapes, processing_fn = metric_tensor(tape, *args, **kwargs)

        def processing_fn_multiply(res):
            res = qml.execute(res, device=device)
            return 4 * processing_fn(res)
        return (tapes, processing_fn_multiply)
    res = adjoint_metric_tensor(tape, *args, **kwargs)

    def processing_fn_multiply(r):
        r = qml.math.stack(r)
        return 4 * r
    return (res, processing_fn_multiply)