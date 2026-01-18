from copy import copy
from typing import Any, Dict, Optional, Sequence, Callable
from pennylane import apply, adjoint
from pennylane.math import mean, shape, round
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms import transform
import pennylane as qml
@transform
def fold_global(tape: QuantumTape, scale_factor) -> (Sequence[QuantumTape], Callable):
    """Differentiable circuit folding of the global unitary ``circuit``.

    For a unitary circuit :math:`U = L_d .. L_1`, where :math:`L_i` can be either a gate or layer, ``fold_global`` constructs

    .. math:: \\text{fold_global}(U) = U (U^\\dagger U)^n (L^\\dagger_d L^\\dagger_{d-1} .. L^\\dagger_s) (L_s .. L_d)

    where :math:`n = \\lfloor (\\lambda - 1)/2 \\rfloor` and :math:`s = \\lfloor \\left(\\lambda - 1 \\right) (d/2) \\rfloor` are determined via the ``scale_factor`` :math:`=\\lambda`.
    The purpose of folding is to artificially increase the noise for zero noise extrapolation, see :func:`~.pennylane.transforms.mitigate_with_zne`.

    Args:
        tape (QNode or QuantumTape): the quantum circuit to be folded
        scale_factor (float): Scale factor :math:`\\lambda` determining :math:`n` and :math:`s`

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]: The folded circuit as described in :func:`qml.transform <pennylane.transform>`.

    .. seealso:: :func:`~.pennylane.transforms.mitigate_with_zne`; This function is analogous to the implementation in ``mitiq``  `mitiq.zne.scaling.fold_global <https://mitiq.readthedocs.io/en/v.0.1a2/apidoc.html?highlight=global_folding#mitiq.zne.scaling.fold_global>`_.

    **Example**

    Let us look at the following circuit.

    .. code-block:: python

        x = np.arange(6)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.RZ(x[2], wires=2)
            qml.CNOT(wires=(0,1))
            qml.CNOT(wires=(1,2))
            qml.RX(x[3], wires=0)
            qml.RY(x[4], wires=1)
            qml.RZ(x[5], wires=2)
            return qml.expval(qml.Z(0) @ qml.Z(1) @ qml.Z(2))


    Setting ``scale_factor=1`` does not affect the circuit:

    >>> folded = qml.transforms.fold_global(circuit, 1)
    >>> print(qml.draw(folded)(x))
    0: ──RX(0.0)─╭●──RX(3.0)──────────┤ ╭<Z@Z@Z>
    1: ──RY(1.0)─╰X─╭●────────RY(4.0)─┤ ├<Z@Z@Z>
    2: ──RZ(2.0)────╰X────────RZ(5.0)─┤ ╰<Z@Z@Z>

    Setting ``scale_factor=2`` results in the partially folded circuit :math:`U (L^\\dagger_d L^\\dagger_{d-1} .. L^\\dagger_s) (L_s .. L_d)`
    with :math:`s = \\lfloor \\left(1 \\mod 2 \\right) d/2 \\rfloor = 4` since the circuit is composed of :math:`d=8` gates.

    >>> folded = qml.transforms.fold_global(circuit, 2)
    >>> print(qml.draw(folded)(x))
    0: ──RX(0.0)─╭●──RX(3.0)──RX(3.0)†──RX(3.0)──────────────────┤ ╭<Z@Z@Z>
    1: ──RY(1.0)─╰X─╭●────────RY(4.0)───RY(4.0)†─╭●──╭●──RY(4.0)─┤ ├<Z@Z@Z>
    2: ──RZ(2.0)────╰X────────RZ(5.0)───RZ(5.0)†─╰X†─╰X──RZ(5.0)─┤ ╰<Z@Z@Z>

    Setting ``scale_factor=3`` results in the folded circuit :math:`U (U^\\dagger U)`.

    >>> folded = qml.transforms.fold_global(circuit, 3)
    >>> print(qml.draw(folded)(x))
    0: ──RX(0.0)─╭●──RX(3.0)──RX(3.0)†───────────────╭●─────────RX(0.0)†──RX(0.0)─╭●──RX(3.0)──────────┤╭<Z@Z@Z>
    1: ──RY(1.0)─╰X─╭●────────RY(4.0)───RY(4.0)†─╭●──╰X†────────RY(1.0)†──RY(1.0)─╰X─╭●────────RY(4.0)─┤├<Z@Z@Z>
    2: ──RZ(2.0)────╰X────────RZ(5.0)───RZ(5.0)†─╰X†──RZ(2.0)†──RZ(2.0)──────────────╰X────────RZ(5.0)─┤╰<Z@Z@Z>

    .. note::

        Circuits are treated as lists of operations. Since the ordering of that list is ambiguous, so is its folding.
        This can be seen exemplarily for two equivalent unitaries :math:`U1 = X(0) Y(0) X(1) Y(1)` and :math:`U2 = X(0) X(1) Y(0) Y(1)`.
        The folded circuits according to ``scale_factor=2`` would be :math:`U1 (X(0) Y(0) Y(0) X(0))` and :math:`U2 (X(0) X(1) X(1) X(0))`, respectively.
        So even though :math:`U1` and :math:`U2` are describing the same quantum circuit, the ambiguity in their ordering as a list yields two differently folded circuits.

    .. details::

        The main purpose of folding is for zero noise extrapolation (ZNE). PennyLane provides a differentiable transform :func:`~.pennylane.transforms.mitigate_with_zne`
        that allows you to perform ZNE as a black box. If you want more control and `see` the extrapolation, you can follow the logic of the following example.

        We start by setting up a noisy device using the mixed state simulator and a noise channel.

        .. code-block:: python

            n_wires = 4

            # Describe noise
            noise_gate = qml.DepolarizingChannel
            noise_strength = 0.05

            # Load devices
            dev_ideal = qml.device("default.mixed", wires=n_wires)
            dev_noisy = qml.transforms.insert(noise_gate, noise_strength)(dev_ideal)

            x = np.arange(6)

            H = 1.*qml.X(0) @ qml.X(1) + 1.*qml.X(1) @ qml.X(2)

            def circuit(x):
                qml.RY(x[0], wires=0)
                qml.RY(x[1], wires=1)
                qml.RY(x[2], wires=2)
                qml.CNOT(wires=(0,1))
                qml.CNOT(wires=(1,2))
                qml.RY(x[3], wires=0)
                qml.RY(x[4], wires=1)
                qml.RY(x[5], wires=2)
                return qml.expval(H)

            qnode_ideal = qml.QNode(circuit, dev_ideal)
            qnode_noisy = qml.QNode(circuit, dev_noisy)

        We can then create folded versions of the noisy qnode and execute them for different scaling factors.

        >>> scale_factors = [1., 2., 3.]
        >>> folded_res = [qml.transforms.fold_global(qnode_noisy, lambda_)(x) for lambda_ in scale_factors]

        We want to later compare the ZNE with the ideal result.

        >>> ideal_res = qnode_ideal(x)

        ZNE is, as the name suggests, an extrapolation in the noise to zero. The underlyding assumption is that the level of noise is proportional to the scaling factor
        by artificially increasing the circuit depth. We can perform a polynomial fit using ``numpy`` functions. Note that internally in :func:`~.pennylane.transforms.mitigate_with_zne`
        a differentiable polynomial fit function :func:`~.pennylane.transforms.poly_extrapolate` is used.

        >>> # coefficients are ordered like coeffs[0] * x**2 + coeffs[1] * x + coeffs[0]
        >>> coeffs = np.polyfit(scale_factors, folded_res, 2)
        >>> zne_res = coeffs[-1]

        We used a polynomial fit of ``order=2``. Using ``order=len(scale_factors) -1`` is also referred to as Richardson extrapolation and implemented in :func:`~.pennylane.transforms.richardson_extrapolate`.
        We can now visualize our fit to see how close we get to the ideal result with this mitigation technique.

        .. code-block:: python

            x_fit = np.linspace(0, 3, 20)
            y_fit = np.poly1d(coeffs)(x_fit)

            plt.plot(scale_factors, folded_res, "x--", label="folded")
            plt.plot(0, ideal_res, "X", label="ideal res")
            plt.plot(0, zne_res, "X", label="ZNE res", color="tab:red")
            plt.plot(x_fit, y_fit, label="fit", color="tab:red", alpha=0.5)
            plt.legend()

        .. figure:: ../../_static/fold_global_zne_by-hand.png
            :align: center
            :width: 60%
            :target: javascript:void(0);


    """
    return ([fold_global_tape(tape, scale_factor)], lambda x: x[0])