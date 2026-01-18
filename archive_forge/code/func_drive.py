from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from .parametrized_hamiltonian import ParametrizedHamiltonian
def drive(amplitude, phase, wires):
    """Returns a :class:`ParametrizedHamiltonian` representing the action of a driving electromagnetic
    field with a set of qubits.

    .. math::
        \\frac{1}{2} \\sum_{j \\in \\text{wires}} \\Omega(t) \\left(e^{i \\phi(t)} \\sigma^+_j + e^{-i \\phi(t)} \\sigma^-_j \\right)

    where :math:`\\Omega` and :math:`\\phi` correspond to the amplitude and phase of the
    electromagnetic driving field and :math:`j` corresponds to the wire index. We are describing the Hamiltonian
    in terms of ladder operators :math:`\\sigma^\\pm = \\frac{1}{2}(\\sigma_x \\pm i \\sigma_y)`. Note that depending on the
    hardware realization (neutral atoms, superconducting qubits), there are different conventions and notations.
    E.g., for superconducting qubits it is common to describe the exponent of the phase factor as :math:`\\exp(i(\\phi(t) + \\nu t))`, where :math:`\\nu` is the
    drive frequency. We describe their relations in the theoretical background section below.

    Common hardware systems are superconducting qubits and neutral atoms. The electromagnetic field of the drive is
    realized by microwave and laser fields, respectively, operating at very different wavelengths.
    To avoid nummerical problems due to using both very large and very small numbers, it is advisable to match
    the order of magnitudes of frequency and time arguments.
    Read the usage details for more information on how to choose :math:`\\Omega` and :math:`\\phi`.

    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude of an
            electromagnetic field
        phase (Union[float, Callable]): float or callable returning the phase (in radians) of the electromagnetic field
        wires (Union[int, List[int]]): integer or list containing wire values for the qubits that
            the electromagnetic field acts on

    Returns:
        ParametrizedHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the action of the electromagnetic field
        on the qubits.

    .. seealso::

        :func:`~.rydberg_interaction`, :class:`~.ParametrizedHamiltonian`, :class:`~.ParametrizedEvolution`
        and :func:`~.evolve`

    **Example**

    We create a Hamiltonian describing an electromagnetic field acting on 4 qubits with a fixed
    phase, as well as a parametrized, time-dependent amplitude. The Hamiltonian includes an interaction term for
    inter-qubit interactions.

    .. code-block:: python3

        wires = [0, 1, 2, 3]
        H_int = sum([qml.X(i) @ qml.X((i+1)%len(wires)) for i in wires])

        amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
        phase = jnp.pi / 2
        H_d = qml.pulse.drive(amplitude, phase, wires)

    >>> H_int
    (1) [X0 X1]
    + (1) [X1 X2]
    + (1) [X2 X3]
    + (1) [X3 X0]
    >>> H_d
    HardwareHamiltonian:: terms=2

    The terms of the drive Hamiltonian ``H_d`` correspond to the two terms
    :math:`\\Omega e^{i \\phi(t)} \\sigma^+_j + \\Omega e^{-i \\phi(t)} \\sigma^-_j`,
    describing a drive between the ground and excited states.
    In this case, the drive term corresponds to a global drive, as it acts on all 4 wires of
    the device.

    The full Hamiltonian can be evaluated:

    .. code-block:: python3

        dev = qml.device("default.qubit.jax", wires=wires)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H_int + H_d)(params, t=[0, 10])
            return qml.expval(qml.Z(0))

    >>> params = [2.4]
    >>> circuit(params)
    Array(0.32495208, dtype=float64)
    >>> jax.grad(circuit)(params)
    [Array(1.31956098, dtype=float64)]

    We can also create a Hamiltonian with multiple local drives. The following circuit corresponds to the
    evolution where an additional local drive that changes in time is acting on wires ``[0, 1]`` is added to the Hamiltonian:

    .. code-block:: python3

        amplitude_local = lambda p, t: p[0] * jnp.sin(2 * jnp.pi * t) + p[1]
        phase_local = lambda p, t: p * jnp.exp(-0.25 * t)
        H_local = qml.pulse.drive(amplitude_local, phase_local, [0, 1])

        H = H_int + H_d + H_local

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit_local(params):
            qml.evolve(H)(params, t=[0, 10])
            return qml.expval(qml.Z(0))

        p_global = 2.4
        p_amp = [1.3, -2.0]
        p_phase = 0.5
        params = (p_global, p_amp, p_phase)

    >>> circuit_local(params)
    Array(-0.5334795, dtype=float64)
    >>> jax.grad(circuit_local)(params)
    (Array(0.01654573, dtype=float64),
     [Array(-0.04422795, dtype=float64, weak_type=True),
      Array(-0.51375441, dtype=float64, weak_type=True)],
     Array(0.21901967, dtype=float64))

    .. details::
        :title: Theoretical background
        :href: theory

        Depending on the community and field it is often common to write the driving field Hamiltonian as

        .. math::
            H = \\frac{1}{2} \\Omega(t) \\sum_{j \\in \\text{wires}} \\left(e^{i (\\phi(t) + \\nu t)} \\sigma^+_j + e^{-i (\\phi(t) + \\nu t)} \\sigma^-_j \\right)
            + \\omega_q \\sum_{j \\in \\text{wires}} \\sigma^z_j,

        with amplitude :math:`\\Omega`, phase :math:`\\phi` and drive frequency :math:`\\nu` of the electromagnetic field, as well as the qubit frequency :math:`\\omega_q`.
        We can move to the rotating frame of the driving field by applying :math:`U = e^{-i\\nu t \\sigma^z}` which yields the new Hamiltonian

        .. math::
            H = \\frac{1}{2} \\Omega(t) \\sum_{j \\in \\text{wires}} \\left(e^{i \\phi(t)} \\sigma^+_j + e^{-i \\phi(t)} \\sigma^-_j \\right)
            - (\\nu - \\omega_q) \\sum_{j \\in \\text{wires}} \\sigma^z_j

        The latter formulation is more common in neutral atom systems where we define the detuning from the atomic energy gap
        as :math:`\\Delta = \\nu - \\omega_q`. This is because here all atoms have the same energy gap, whereas for superconducting
        qubits that is typically not the case.
        Note that a potential anharmonicity term, as is common for transmon systems when taking into account higher energy
        levels, is unaffected by this transformation.

        Further, note that the factor :math:`\\frac{1}{2}` is a matter of convention. We keep it for ``drive()`` as well as :func:`~.rydberg_drive`,
        but ommit it in :func:`~.transmon_drive`, as is common in the respective fields.

    .. details::
        **Neutral Atom Rydberg systems**

        In neutral atom systems for quantum computation and quantum simulation, a Rydberg transition is driven by an optical laser that is close to the transition's resonant frequency (with a potential detuning with regards to the resonant frequency on the order of MHz).
        The interaction between different atoms is given by the :func:`rydberg_interaction`, for which we pass the atomic coordinates (in µm),
        here arranged in a square of length :math:`4 \\mu m`.

        .. code-block:: python3

            atom_coordinates = [[0, 0], [0, 4], [4, 0], [4, 4]]
            wires = [1, 2, 3, 4]
            assert len(wires) == len(atom_coordinates)
            H_i = qml.pulse.rydberg_interaction(atom_coordinates, wires)

        We can now simulate driving those atoms with an oscillating amplitude :math:`\\Omega` that is trainable, for a duration of :math:`10 \\mu s`.

        .. code-block:: python3

            amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
            phase = jnp.pi / 2

            H_d = qml.pulse.drive(amplitude, phase, wires)

            # detuning term
            H_z = qml.dot([-3*np.pi/4]*len(wires), [qml.Z(i) for i in wires])


        The total Hamiltonian of that evolution is given by

        .. math::
            \\frac{1}{2} p \\sin(\\pi t) \\sum_{j \\in \\text{wires}} \\left(e^{i \\pi/2} \\sigma^+_j + e^{-i \\pi/2} \\sigma^-_j \\right) -
            \\frac{3 \\pi}{4} \\sum_{j \\in \\text{wires}} \\sigma^z_j + \\sum_{k<\\ell} V_{k \\ell} n_k n_\\ell

        and can be executed and differentiated via the following code.

        .. code-block:: python3

            dev = qml.device("default.qubit.jax", wires=wires)
            @qml.qnode(dev, interface="jax")
            def circuit(params):
                qml.evolve(H_i + H_z + H_d)(params, t=[0, 10])
                return qml.expval(qml.Z(1))

        >>> params = [2.4]
        >>> circuit(params)
        Array(0.6962041, dtype=float64)
        >>> jax.grad(circuit)(params)
        [Array(1.75825695, dtype=float64)]
    """
    wires = Wires(wires)
    coeffs = [amplitude_and_phase(qml.math.cos, amplitude, phase), amplitude_and_phase(qml.math.sin, amplitude, phase)]
    drive_x_term = 0.5 * sum((qml.X(wire) for wire in wires))
    drive_y_term = -0.5 * sum((qml.Y(wire) for wire in wires))
    observables = [drive_x_term, drive_y_term]
    return HardwareHamiltonian(coeffs, observables, _reorder_parameters)