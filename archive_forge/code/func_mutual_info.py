from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
@partial(transform, final_transform=True)
def mutual_info(tape: QuantumTape, wires0: Sequence[int], wires1: Sequence[int], base: float=None, **kwargs) -> (Sequence[QuantumTape], Callable):
    """Compute the mutual information from a :class:`.QuantumTape` returning a :func:`~pennylane.state`:

    .. math::

        I(A, B) = S(\\rho^A) + S(\\rho^B) - S(\\rho^{AB})

    where :math:`S` is the von Neumann entropy.

    The mutual information is a measure of correlation between two subsystems.
    More specifically, it quantifies the amount of information obtained about
    one system by measuring the other system.

    Args:
        qnode (QNode or QuantumTape or Callable): A quantum circuit returning a :func:`~pennylane.state`.
        wires0 (Sequence(int)): List of wires in the first subsystem.
        wires1 (Sequence(int)): List of wires in the second subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the mutual information in the form of a tensor.

    **Example**

    It is possible to obtain the mutual information of two subsystems from a
    :class:`.QNode` returning a :func:`~pennylane.state`.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

    >>> mutual_info_circuit = qinfo.mutual_info(circuit, wires0=[0], wires1=[1])
    >>> mutual_info_circuit(np.pi/2)
    1.3862943611198906
    >>> x = np.array(0.4, requires_grad=True)
    >>> mutual_info_circuit(x)
    0.3325090393262875
    >>> qml.grad(mutual_info_circuit)(np.array(0.4, requires_grad=True))
    tensor(1.24300677, requires_grad=True)

    .. seealso:: :func:`~.qinfo.vn_entropy`, :func:`pennylane.math.mutual_info` and :func:`pennylane.mutual_info`
    """
    all_wires = kwargs.get('device_wires', tape.wires)
    wire_map = {w: i for i, w in enumerate(all_wires)}
    indices0 = [wire_map[w] for w in wires0]
    indices1 = [wire_map[w] for w in wires1]
    measurements = tape.measurements
    if len(measurements) != 1 or not isinstance(measurements[0], StateMP):
        raise ValueError('The qfunc return type needs to be a state.')

    def processing_fn(res):
        device = kwargs.get('device', None)
        c_dtype = getattr(device, 'C_DTYPE', 'complex128')
        density_matrix = res[0] if isinstance(measurements[0], DensityMatrixMP) or isinstance(device, DefaultMixed) else qml.math.dm_from_state_vector(res[0], c_dtype=c_dtype)
        entropy = qml.math.mutual_info(density_matrix, indices0, indices1, base=base, c_dtype=c_dtype)
        return entropy
    return ([tape], processing_fn)