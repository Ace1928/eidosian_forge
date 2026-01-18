import warnings
from functools import reduce, partial
from itertools import product
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ClassicalShadowMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane import transform
@partial(transform, final_transform=True)
def shadow_expval(tape: QuantumTape, H, k=1) -> (Sequence[QuantumTape], Callable):
    """Transform a circuit returning a classical shadow into one that returns
    the approximate expectation values in a differentiable manner.

    See :func:`~.pennylane.shadow_expval` for more usage details.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        H (:class:`~.pennylane.Observable` or list[:class:`~.pennylane.Observable`]): Observables
            for which to compute the expectation values
        k (int): k (int): Number of equal parts to split the shadow's measurements to compute
            the median of means. ``k=1`` corresponds to simply taking the mean over all measurements.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the expectation value estimates for each observable in the form of a tensor.

    **Example**

    .. code-block:: python3

        H = qml.Z(0) @ qml.Z(1)
        dev = qml.device("default.qubit", wires=2, shots=10000)

        @partial(qml.shadows.shadow_expval, H, k=1)
        @qml.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=0)
            return qml.classical_shadow(wires=[0, 1])

    >>> x = np.array(1.2)
    >>> circuit(x)
    [array(0.3528)]
    >>> qml.grad(circuit)(x)
    -0.9323999999999998
    """
    tapes, _ = _replace_obs(tape, qml.shadow_expval, H, k=k)

    def post_processing_fn(res):
        return res
    return (tapes, post_processing_fn)