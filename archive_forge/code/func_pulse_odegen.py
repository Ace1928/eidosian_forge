from typing import Callable, Sequence
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.pulse import ParametrizedEvolution
from pennylane.ops.qubit.special_unitary import pauli_basis_strings, _pauli_decompose
from pennylane import transform
from .parameter_shift import _make_zero_rep
from .pulse_gradient import _assert_has_jax, raise_pulse_diff_on_qnode
from .gradient_transform import (
@partial(transform, final_transform=True)
def pulse_odegen(tape: qml.tape.QuantumTape, argnum=None, atol=1e-07) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Transform a circuit to compute the pulse generator parameter-shift gradient of pulses
    in a pulse program with respect to their inputs.
    This method combines automatic differentiation of few-qubit operations with
    hardware-compatible shift rules.
    It allows for the evaluation of parameter-shift gradients for many-qubit pulse programs
    on hardware, with the limitation that the individual pulses must be acting on a
    sufficiently small number of qubits.

    For this differentiation method, the unitary matrix :math:`U` of a pulse gate and its derivative
    :math:`\\partial_k U` are computed classically with an autodiff framework.
    From :math:`\\partial_k U` and :math:`U` we can deduce the so-called effective generators
    :math:`\\Omega_{k}` assuming the form

    .. math:: \\partial_k U = U \\Omega_k.

    These effective generators are then decomposed into the Pauli basis and the
    standard parameter-shift rule is used to evaluate the derivatives of the pulse program
    in this basis.

    To this end, shifted ``PauliRot`` operations are inserted in the program.
    Finally, the Pauli basis derivatives are recombined into the gradient
    of the pulse program with respect to its original parameters.
    See the theoretical background section below for more details.

    Args:
        tape (QuantumTape): quantum circuit to differentiate
        argnum (int or list[int] or None): Trainable tape parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned. Note that the indices are with respect to
            the list of trainable parameters.
        atol (float): Precision parameter used to truncate the Pauli basis coefficients
            of the effective generators. Coefficients ``x`` satisfying
            ``qml.math.isclose(x, 0., atol=atol, rtol=0) == True`` are neglected.

    Returns:
        tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the Jacobian in the form of a tensor, a tuple, or a nested tuple depending upon the nesting
        structure of measurements in the original circuit.

    .. note::

        This function requires the JAX interface and does not work with other autodiff interfaces
        commonly encountered with PennyLane.
        In addition, this transform is only JIT-compatible with pulses that only have scalar
        parameters.

    .. warning::

        This transform may not be applied directly to QNodes. Use JAX entrypoints
        (``jax.grad``, ``jax.jacobian``, ...) instead or apply the transform on the tape
        level. Also see the examples below.

    **Example**

    Consider the parameterized Hamiltonian
    :math:`\\theta_0 Y_{0}+f(\\boldsymbol{\\theta_1}, t) Y_{1} + \\theta_2 Z_{0}X_{1}`
    with parameters :math:`\\theta_0 = \\frac{1}{5}`,
    :math:`\\boldsymbol{\\theta_1}=\\left(\\frac{3}{5}, \\frac{1}{5}\\right)^{T}` and
    :math:`\\theta_2 = \\frac{2}{5}`, the time-dependent function
    :math:`f(\\boldsymbol{\\theta_1}, t) = \\theta_{1,0} t + \\theta_{1,1}`
    as well as a time interval :math:`t=\\left[\\frac{1}{10}, \\frac{9}{10}\\right]`.

    .. code-block:: python

        jax.config.update("jax_enable_x64", True)
        H = (
            qml.pulse.constant * qml.Y(0)
            + jnp.polyval * qml.Y(1)
            + qml.pulse.constant * (qml.Z(0) @ qml.X(1))
        )
        params = [jnp.array(0.2), jnp.array([0.6, 0.2]), jnp.array(0.4)]
        t = [0.1, 0.9]

    For simplicity, consider a pulse program consisting of this single pulse and a
    measurement of the expectation value of :math:`X_{0}`.

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, interface="jax", diff_method=qml.gradients.pulse_odegen)
        def circuit(params):
            op = qml.evolve(H)(params, t)
            return qml.expval(qml.X(0))

    We registered the ``QNode`` to be differentiated with the ``pulse_odegen`` method.
    This allows us to simply differentiate it with ``jax.grad``, which internally
    makes use of the pulse generator parameter-shift method.

    >>> jax.grad(circuit)(params)
    [Array(1.41897932, dtype=float64, weak_type=True),
     Array([0.00164913, 0.00284788], dtype=float64),
     Array(-0.09984584, dtype=float64, weak_type=True)]

    Alternatively, we may apply the transform to the tape of the pulse program, obtaining
    the tapes with inserted ``PauliRot`` gates together with the post-processing function:

    >>> circuit.construct((params,), {}) # Build the tape of the circuit.
    >>> tapes, fun = qml.gradients.pulse_odegen(circuit.tape, argnums=[0, 1, 2])
    >>> len(tapes)
    12

    Why are there :math:`12` tapes?
    Consider the terms in the time-dependent pulse Hamiltonian: :math:`\\{Y_0, Y_1, Z_0X_1\\}`.
    Via the Lie bracket, which is just the standard matrix commutator, they
    generate an algebra, the so-called *dynamical Lie algebra (DLA)* of the pulse.
    In order to find all Pauli words that occur in the DLA, we need to (recursively)
    calculate all possible commutators between the three words above and their
    commutators. For the three words above, we obtain three additional words:

    .. math::

        [Y_0, Z_0X_1] &\\propto X_0X_1 \\\\
        [Y_1, Z_0X_1] &\\propto Z_0Z_1 \\\\
        [Y_0, Z_0Z_1] &\\propto X_0Z_1

    All other commutators result in expressions proportional to one of the six Pauli words.
    For each of these six words, we need to compute the standard parameter-shift rule
    requiring two shifted circuits, which yields :math:`12` tapes.

    We may inspect one of the tapes, which differs from the original tape by the inserted
    rotation gate ``"RIY"``, i.e. a ``PauliRot(np.pi/2, "IY", wires=[0, 1])`` gate.
    Note that the order of the tapes follows lexicographical ordering of the inserted
    Pauli rotations, so that :math:`Y_1` is the first of the six words.

    >>> print(qml.drawer.tape_text(tapes[0]))
    0: ─╭RIY─╭ParametrizedEvolution─┤  <X>
    1: ─╰RIY─╰ParametrizedEvolution─┤

    Executing the tapes and applying the post-processing function to the results then
    yields the gradient:

    >>> fun(qml.execute(tapes, dev))
    (Array(1.41897932, dtype=float64),
     Array([0.00164913, 0.00284788], dtype=float64),
     Array(-0.09984584, dtype=float64))

    .. note::

        For pulse Hamiltonians with complex generating terms and few parameters,
        the decomposition approach taken in this method may incur more
        (quantum and classical) computational cost than strictly necessary.

    .. details::
        :title: Theoretical background
        :href: theory

        The pulse generator parameter-shift gradient method makes use of the *effective generator*
        of a pulse for given parameters and duration. Consider the parametrized Hamiltonian

        .. math::

            H(\\boldsymbol{\\theta}, t) = \\sum_{k=1}^K f_k(\\boldsymbol{\\theta}, t) H_k

        where the Hamiltonian terms :math:`\\{H_k\\}` are constant and the :math:`\\{f_k\\}` are
        parametrized time-dependent functions depending on the parameters
        :math:`\\boldsymbol{\\theta}`.
        The unitary time evolution operator associated with :math:`H` is the solution to the
        Schrödinger equation

        .. math::

            \\frac{\\mathrm{d} U}{\\mathrm{d} t}(t) =
            -i H(\\boldsymbol{\\theta}, t) U(t), \\quad U(0) = \\mathbb{I}

        For a fixed time interval :math:`[t_0, t_1]`, we associate a matrix function
        :math:`U(\\boldsymbol{\\theta})` with the unitary evolution.
        To compute the pulse generator parameter-shift gradient, we are interested in the partial
        derivatives of this matrix function, usually with respect to the parameters
        :math:`\\boldsymbol{\\theta}`. Provided that :math:`H` does not act on too many qubits,
        or that we have an alternative sparse representation of
        :math:`U(\\boldsymbol{\\theta})`, we may compute these partial derivatives

        .. math::

            \\frac{\\partial U(\\boldsymbol{\\theta})}{\\partial \\theta_{k}}

        classically via automatic differentiation, where :math:`\\theta_{k}` is
        the :math:`k`\\ -th (scalar) parameter in :math:`\\boldsymbol{\\theta}`.

        Now, due to the compactness of the groups :math:`\\mathrm{SU}(N)`\\ , we know that
        for each :math:`\\theta_{k}` there is an *effective generator* :math:`\\Omega_{k}`
        such that

        .. math::

            \\frac{\\partial U(\\boldsymbol{\\theta})}{\\partial \\theta_{k}} =
            U(\\boldsymbol{\\theta})\\Omega_{k}.

        Given that we can compute the left-hand side expression as well as the matrix
        for :math:`U` itself, we can compute :math:`\\Omega_{k}` for all parameters
        :math:`\\theta_{k}`.
        In addition, we may decompose these generators into Pauli words:

        .. math::

            \\Omega_{k} = \\sum_{\\ell=1}^{L} \\omega_{k}^{(\\ell)} P_{\\ell}

        The coefficients :math:`\\omega_{k}^{(\\ell)}` of the generators can be computed
        by decomposing the anti-Hermitian matrix :math:`\\Omega_{k}` into the Pauli
        basis and only keeping the non-vanishing terms. This is possible via a tensor
        contraction with the full Pauli basis (or alternative, more efficient methods):

        .. math::

            \\omega_{k}^{(\\ell)} = \\frac{1}{2^N}\\mathrm{Tr}\\left[P_\\ell \\Omega_{k}\\right]

        where :math:`N` is the number of qubits and :math:`\\ell = 1, .. , L` the Pauli word index.
        The number of non-zero Pauli words :math:`L` is typically equal to the dimension of the dynamical Lie algebra
        (can be lower if coefficients happen to be zero)
        and at most :math:`4^N-1`.

        Thus far, we discussed the derivative of the time evolution, or pulse.
        Now, consider an objective function that is based on measuring an expectation
        value after executing a pulse program:

        .. math::
            C(\\boldsymbol{\\theta})=
            \\langle\\psi_0|U(\\boldsymbol{\\theta})^\\dagger B
            U(\\boldsymbol{\\theta}) |\\psi_0\\rangle

        Using the derivative of :math:`U` and the decomposition of the effective
        generator :math:`\\Omega_k` above, we calculate the partial derivative of
        :math:`C`:

        .. math::

            \\frac{\\partial C}{\\partial \\theta_{k}} (\\boldsymbol{\\theta})&=
            \\langle\\psi_0|\\left[U^\\dagger B U, \\Omega_{k}\\right]|\\psi_0\\rangle\\\\
            &=\\sum_{\\ell=1}^L \\omega_{k}^{(\\ell)}
            \\langle\\psi_0|\\left[U^\\dagger B U, P_\\ell \\right]|\\psi_0\\rangle\\\\
            &=\\sum_{\\ell=1}^L \\tilde\\omega_{k}^{(\\ell)}
            \\langle\\psi_0|\\left[U^\\dagger B U, -\\frac{i}{2}P_\\ell \\right]|\\psi_0\\rangle\\\\
            &=\\sum_{\\ell=1}^L \\tilde\\omega_{k}^{(\\ell)}
            \\frac{\\mathrm{d}}{\\mathrm{d}x}
            \\langle\\psi_0|\\exp\\left(i\\frac{x}{2}P_\\ell \\right)U^\\dagger B
            U\\exp\\left(-i\\frac{x}{2}P_\\ell \\right)|\\psi_0\\rangle\\large|_{x=0}\\\\
            &=\\sum_{\\ell=1}^L \\tilde\\omega_{k}^{(\\ell)}
            \\frac{\\mathrm{d}}{\\mathrm{d}x} C_\\ell(x)\\large|_{x=0}

        where we skipped the argument :math:`\\boldsymbol{\\theta}` of :math:`U` for readability
        and introduced the modified coefficients
        :math:`\\tilde\\omega_{k}^{(\\ell)}=2i\\omega_{k}^{(\\ell)}`.
        In the second to last step, we rewrote the commutator of :math:`U^\\dagger BU` and
        :math:`\\frac{i}{2}P_\\ell` as the derivative (at zero) of a modified cost function
        :math:`C_\\ell(x)` that executes a Pauli rotation about :math:`-i\\frac{x}{2}P_\\ell`
        before the parametrized time evolution. Here, the variable :math:`x` is just a
        convenient way to write the modified cost function. Note that its derivative with
        respect to :math:`x` can be computed with the standard two-term parameter-shift
        rule for Pauli rotation gates, i.e.

        .. math::

            \\frac{\\mathrm{d}}{\\mathrm{d}x} C_\\ell(x) {\\large|}_{x=0} = \\frac{1}{2} \\left(C_\\ell(\\pi/2) - C_\\ell(-\\pi/2)\\right)

        with :math:`C_\\ell(x) = \\langle\\psi_0|e^{i\\frac{x}{2}P_\\ell} U^\\dagger B U e^{-i\\frac{x}{2}P_\\ell} |\\psi_0\\rangle`.

        **Caching**

        Considering the derivation above, we notice that the same modified cost function
        :math:`C_\\ell(x)` may appear in the derivatives of distinct parameters
        :math:`\\theta_k` and :math:`\\theta_m`, because they are shared by two terms in the pulse Hamiltonian.
        In order to not evaluate the same
        modified quantum circuit derivatives multiple times, we use an internal
        cache that avoids repeated creation of the same parameter-shifted circuits.
        In addition, all modified cost functions :math:`C_\\ell` that would be multiplied
        with a vanishing coefficient :math:`\\tilde\\omega_{k}^{(\\ell)}` *for all* :math:`k`
        are skipped altogether.
        This approach requires a few additional classical coprocessing steps but allows
        us to save quantum resources in many relevant pulse programs.

    """
    transform_name = 'pulse generator parameter-shift'
    _assert_has_jax(transform_name)
    assert_no_state_returns(tape.measurements, transform_name)
    assert_no_variance(tape.measurements, transform_name)
    assert_no_tape_batching(tape, transform_name)
    if argnum is None and (not tape.trainable_params):
        return _no_trainable_grad(tape)
    trainable_params = choose_trainable_params(tape, argnum)
    diff_methods = find_and_validate_gradient_methods(tape, 'analytic', trainable_params)
    if all((g == '0' for g in diff_methods.values())):
        return _all_zero_grad(tape)
    argnum = [i for i, dm in diff_methods.items() if dm == 'A']
    return _expval_pulse_odegen(tape, argnum, atol)