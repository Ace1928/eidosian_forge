from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
def relative_entropy(qnode0, qnode1, wires0, wires1):
    """
    Compute the relative entropy for two :class:`.QNode` returning a :func:`~pennylane.state` (a state can be a state vector
    or a density matrix, depending on the device) acting on quantum systems with the same size.

    .. math::
        S(\\rho\\,\\|\\,\\sigma)=-\\text{Tr}(\\rho\\log\\sigma)-S(\\rho)=\\text{Tr}(\\rho\\log\\rho)-\\text{Tr}(\\rho\\log\\sigma)
        =\\text{Tr}(\\rho(\\log\\rho-\\log\\sigma))

    Roughly speaking, quantum relative entropy is a measure of distinguishability between two
    quantum states. It is the quantum mechanical analog of relative entropy.

    Args:
        qnode0 (QNode): A :class:`.QNode` returning a :func:`~pennylane.state`.
        qnode1 (QNode): A :class:`.QNode` returning a :func:`~pennylane.state`.
        wires0 (Sequence[int]): the subsystem of the first QNode
        wires1 (Sequence[int]): the subsystem of the second QNode

    Returns:
        func: A function that takes as input the joint arguments of the two QNodes,
        and returns the relative entropy from their output states.

    **Example**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

    The ``qml.qinfo.relative_entropy`` transform can be used to compute the relative
    entropy between the output states of the QNode:

    >>> relative_entropy_circuit = qml.qinfo.relative_entropy(circuit, circuit, wires0=[0], wires1=[0])

    The returned function takes two tuples as input, the first being the arguments to the
    first QNode and the second being the arguments to the second QNode:

    >>> x, y = np.array(0.4), np.array(0.6)
    >>> relative_entropy_circuit((x,), (y,))
    tensor(0.01775001, requires_grad=True)

    This transform is fully differentiable:

    .. code-block:: python

        def wrapper(x, y):
            return relative_entropy_circuit((x,), (y,))

    >>> wrapper(x, y)
    tensor(0.01775001, requires_grad=True)
    >>> qml.grad(wrapper)(x, y)
    (tensor(-0.16458856, requires_grad=True),
     tensor(0.16953273, requires_grad=True))
    """
    if len(wires0) != len(wires1):
        raise qml.QuantumFunctionError('The two states must have the same number of wires.')
    state_qnode0 = qml.qinfo.reduced_dm(qnode0, wires=wires0)
    state_qnode1 = qml.qinfo.reduced_dm(qnode1, wires=wires1)

    def evaluate_relative_entropy(all_args0=None, all_args1=None):
        """Wrapper used for evaluation of the relative entropy between two states computed from
        QNodes. It allows giving the args and kwargs to each :class:`.QNode`.

        Args:
            all_args0 (tuple): Tuple containing the arguments (*args, kwargs) of the first :class:`.QNode`.
            all_args1 (tuple): Tuple containing the arguments (*args, kwargs) of the second :class:`.QNode`.

        Returns:
            float: Relative entropy between two quantum states
        """
        if not isinstance(all_args0, tuple) and all_args0 is not None:
            all_args0 = (all_args0,)
        if not isinstance(all_args1, tuple) and all_args1 is not None:
            all_args1 = (all_args1,)
        if all_args0 is not None:
            if isinstance(all_args0[-1], dict):
                args0 = all_args0[:-1]
                kwargs0 = all_args0[-1]
            else:
                args0 = all_args0
                kwargs0 = {}
            state0 = state_qnode0(*args0, **kwargs0)
        else:
            state0 = state_qnode0()
        if all_args1 is not None:
            if isinstance(all_args1[-1], dict):
                args1 = all_args1[:-1]
                kwargs1 = all_args1[-1]
            else:
                args1 = all_args1
                kwargs1 = {}
            state1 = state_qnode1(*args1, **kwargs1)
        else:
            state1 = state_qnode1()
        return qml.math.relative_entropy(state0, state1)
    return evaluate_relative_entropy