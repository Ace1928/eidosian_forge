from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
@partial(transform, final_transform=True)
def purity(tape: QuantumTape, wires, **kwargs) -> (Sequence[QuantumTape], Callable):
    """Compute the purity of a :class:`~.QuantumTape` returning :func:`~pennylane.state`.

    .. math::
        \\gamma = \\text{Tr}(\\rho^2)

    where :math:`\\rho` is the density matrix. The purity of a normalized quantum state satisfies
    :math:`\\frac{1}{d} \\leq \\gamma \\leq 1`, where :math:`d` is the dimension of the Hilbert space.
    A pure state has a purity of 1.

    It is possible to compute the purity of a sub-system from a given state. To find the purity of
    the overall state, include all wires in the ``wires`` argument.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit object returning a :func:`~pennylane.state`.
        wires (Sequence(int)): List of wires in the considered subsystem.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the purity in the form of a tensor.

    **Example**

    .. code-block:: python

        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev)
        def noisy_circuit(p):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.BitFlip(p, wires=0)
            qml.BitFlip(p, wires=1)
            return qml.state()

        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

    >>> purity(noisy_circuit, wires=[0, 1])(0.2)
    0.5648000000000398
    >>> purity(circuit, wires=[0])(np.pi / 2)
    0.5
    >>> purity(circuit, wires=[0, 1])(np.pi / 2)
    1.0

    .. seealso:: :func:`pennylane.math.purity`
    """
    all_wires = kwargs.get('device_wires', tape.wires)
    wire_map = {w: i for i, w in enumerate(all_wires)}
    indices = [wire_map[w] for w in wires]
    measurements = tape.measurements
    if len(measurements) != 1 or not isinstance(measurements[0], StateMP):
        raise ValueError('The qfunc return type needs to be a state.')

    def processing_fn(res):
        device = kwargs.get('device', None)
        c_dtype = getattr(device, 'C_DTYPE', 'complex128')
        density_matrix = res[0] if isinstance(measurements[0], DensityMatrixMP) or isinstance(device, DefaultMixed) else qml.math.dm_from_state_vector(res[0], c_dtype=c_dtype)
        return qml.math.purity(density_matrix, indices, c_dtype=c_dtype)
    return ([tape], processing_fn)