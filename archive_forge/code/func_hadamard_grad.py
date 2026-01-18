from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.gradients.metric_tensor import _get_aux_wire
from pennylane import transform
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from pennylane.transforms.tape_expand import expand_invalid_trainable_hadamard_gradient
from .gradient_transform import (
@partial(transform, expand_transform=_expand_transform_hadamard, classical_cotransform=_contract_qjac_with_cjac, final_transform=True)
def hadamard_grad(tape: qml.tape.QuantumTape, argnum=None, aux_wire=None, device_wires=None) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Transform a circuit to compute the Hadamard test gradient of all gates
    with respect to their inputs.

    Args:
        tape (QNode or QuantumTape): quantum circuit to differentiate
        argnum (int or list[int] or None): Trainable tape parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned. Note that the indices are with respect to
            the list of trainable parameters.
        aux_wire (pennylane.wires.Wires): Auxiliary wire to be used for the Hadamard tests.
            If ``None`` (the default), a suitable wire is inferred from the wires used in
            the original circuit and ``device_wires``.
        device_wires (pennylane.wires.Wires): Wires of the device that are going to be used for the
            gradient. Facilitates finding a default for ``aux_wire`` if ``aux_wire`` is ``None``.

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.
        Executing this circuit will provide the Jacobian in the form of a tensor, a tuple, or a
        nested tuple depending upon the nesting structure of measurements in the original circuit.

    For a variational evolution :math:`U(\\mathbf{p}) \\vert 0\\rangle` with :math:`N` parameters
    :math:`\\mathbf{p}`, consider the expectation value of an observable :math:`O`:

    .. math::

        f(\\mathbf{p})  = \\langle \\hat{O} \\rangle(\\mathbf{p}) = \\langle 0 \\vert
        U(\\mathbf{p})^\\dagger \\hat{O} U(\\mathbf{p}) \\vert 0\\rangle.


    The gradient of this expectation value can be calculated via the Hadamard test gradient:

    .. math::

        \\frac{\\partial f}{\\partial \\mathbf{p}} = -2 \\Im[\\bra{0} \\hat{O} G \\ket{0}] = i \\left(\\bra{0} \\hat{O} G \\ket{
        0} - \\bra{0} G\\hat{O} \\ket{0}\\right) = -2 \\bra{+}\\bra{0} ctrl-G^{\\dagger} (\\hat{Y} \\otimes \\hat{O}) ctrl-G
        \\ket{+}\\ket{0}

    Here, :math:`G` is the generator of the unitary :math:`U`.

    **Example**

    This transform can be registered directly as the quantum gradient transform
    to use during autodifferentiation:

    >>> import jax
    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev, interface="jax", diff_method="hadamard")
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.Z(0)), qml.var(qml.Z(0))
    >>> params = jax.numpy.array([0.1, 0.2, 0.3])
    >>> jax.jacobian(circuit)(params)
    (Array([-0.38751727, -0.18884793, -0.3835571 ], dtype=float32),
    Array([0.6991687 , 0.34072432, 0.6920237 ], dtype=float32))

    .. details::
        :title: Usage Details

        This gradient transform can be applied directly to :class:`QNode <pennylane.QNode>`
        objects. However, for performance reasons, we recommend providing the gradient transform
        as the ``diff_method`` argument of the QNode decorator, and differentiating with your
        preferred machine learning framework.

        >>> dev = qml.device("default.qubit", wires=2)
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.Z(0))
        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.gradients.hadamard_grad(circuit)(params)
        (tensor([-0.3875172], requires_grad=True),
         tensor([-0.18884787], requires_grad=True),
         tensor([-0.38355704], requires_grad=True))

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> ops = [qml.RX(p, wires=0) for p in params]
        >>> measurements = [qml.expval(qml.Z(0))]
        >>> tape = qml.tape.QuantumTape(ops, measurements)
        >>> gradient_tapes, fn = qml.gradients.hadamard_grad(tape)
        >>> gradient_tapes
        [<QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed.

        Note that ``argnum`` refers to the index of a parameter within the list of trainable
        parameters. For example, if we have:

        >>> tape = qml.tape.QuantumScript(
        ...     [qml.RX(1.2, wires=0), qml.RY(2.3, wires=0), qml.RZ(3.4, wires=0)],
        ...     [qml.expval(qml.Z(0))],
        ...     trainable_params = [1, 2]
        ... )
        >>> qml.gradients.hadamard_grad(tape, argnum=1)

        The code above will differentiate the third parameter rather than the second.

        The output tapes can then be evaluated and post-processed to retrieve the gradient:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(gradient_tapes, dev, None))
        (array(-0.3875172), array(-0.18884787), array(-0.38355704))

        This transform can be registered directly as the quantum gradient transform
        to use during autodifferentiation:

        >>> dev = qml.device("default.qubit", wires=3)
        >>> @qml.qnode(dev, interface="jax", diff_method="hadamard")
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.Z(0))
        >>> params = jax.numpy.array([0.1, 0.2, 0.3])
        >>> jax.jacobian(circuit)(params)
        [-0.3875172  -0.18884787 -0.38355704]

        If you use custom wires on your device, you need to pass an auxiliary wire.

        >>> dev_wires = ("a", "c")
        >>> dev = qml.device("default.qubit", wires=dev_wires)
        >>> @qml.qnode(dev, interface="jax", diff_method="hadamard", aux_wire="c", device_wires=dev_wires)
        >>> def circuit(params):
        ...    qml.RX(params[0], wires="a")
        ...    qml.RY(params[1], wires="a")
        ...    qml.RX(params[2], wires="a")
        ...    return qml.expval(qml.Z("a"))
        >>> params = jax.numpy.array([0.1, 0.2, 0.3])
        >>> jax.jacobian(circuit)(params)
        [-0.3875172  -0.18884787 -0.38355704]

    .. note::

        ``hadamard_grad`` will decompose the operations that are not in the list of supported operations.

        - ``pennylane.RX``
        - ``pennylane.RY``
        - ``pennylane.RZ``
        - ``pennylane.Rot``
        - ``pennylane.PhaseShift``
        - ``pennylane.U1``
        - ``pennylane.CRX``
        - ``pennylane.CRY``
        - ``pennylane.CRZ``
        - ``pennylane.IsingXX``
        - ``pennylane.IsingYY``
        - ``pennylane.IsingZZ``

        The expansion will fail if a suitable decomposition in terms of supported operation is not found.
        The number of trainable parameters may increase due to the decomposition.

    """
    transform_name = 'Hadamard test'
    assert_no_state_returns(tape.measurements, transform_name)
    assert_no_variance(tape.measurements, transform_name)
    assert_no_tape_batching(tape, transform_name)
    if len(tape.measurements) > 1 and tape.shots.has_partitioned_shots:
        raise NotImplementedError('hadamard gradient does not support multiple measurements with partitioned shots.')
    if argnum is None and (not tape.trainable_params):
        return _no_trainable_grad(tape)
    trainable_params = choose_trainable_params(tape, argnum)
    diff_methods = find_and_validate_gradient_methods(tape, 'analytic', trainable_params)
    if all((g == '0' for g in diff_methods.values())):
        return _all_zero_grad(tape)
    argnum = [i for i, dm in diff_methods.items() if dm == 'A']
    aux_wire = _get_aux_wire(aux_wire, tape, device_wires)
    g_tapes, processing_fn = _expval_hadamard_grad(tape, argnum, aux_wire)
    return (g_tapes, processing_fn)