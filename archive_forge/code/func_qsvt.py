import copy
import numpy as np
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.ops import BlockEncode, PCPhase
from pennylane.ops.op_math import adjoint
from pennylane.operation import AnyWires, Operation
def qsvt(A, angles, wires, convention=None):
    """Implements the
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ (QSVT) circuit.

    .. note ::

        :class:`~.BlockEncode` and :class:`~.PCPhase` used in this implementation of QSVT
        are matrix-based operators and well-suited for simulators.
        To implement QSVT with user-defined circuits for the block encoding and
        projector-controlled phase shifts, use the :class:`~.QSVT` template.

    Given a matrix :math:`A`, and a list of angles :math:`\\vec{\\phi}`, this function applies a
    circuit for the quantum singular value transformation using :class:`~.BlockEncode` and
    :class:`~.PCPhase`.

    When the number of angles is even (:math:`d` is odd), the QSVT circuit is defined as:

    .. math::

        U_{QSVT} = \\tilde{\\Pi}_{\\phi_1}U\\left[\\prod^{(d-1)/2}_{k=1}\\Pi_{\\phi_{2k}}U^\\dagger
        \\tilde{\\Pi}_{\\phi_{2k+1}}U\\right]\\Pi_{\\phi_{d+1}},


    and when the number of angles is odd (:math:`d` is even):

    .. math::

        U_{QSVT} = \\left[\\prod^{d/2}_{k=1}\\Pi_{\\phi_{2k-1}}U^\\dagger\\tilde{\\Pi}_{\\phi_{2k}}U\\right]
        \\Pi_{\\phi_{d+1}}.

    Here, :math:`U` denotes a block encoding of :math:`A` via :class:`~.BlockEncode` and
    :math:`\\Pi_\\phi` denotes a projector-controlled phase shift with angle :math:`\\phi`
    via :class:`~.PCPhase`.

    This circuit applies a polynomial transformation (:math:`Poly^{SV}`) to the singular values of
    the block encoded matrix:

    .. math::

        \\begin{align}
             U_{QSVT}(A, \\phi) &=
             \\begin{bmatrix}
                Poly^{SV}(A) & \\cdot \\\\
                \\cdot & \\cdot
            \\end{bmatrix}.
        \\end{align}

    The polynomial transformation is determined by a combination of the block encoding and choice of angles,
    :math:`\\vec{\\phi}`. The convention used by :class:`~.BlockEncode` is commonly refered to as the
    reflection convention or :math:`R` convention. Another equivalent convention for the block encoding is
    the :math:`Wx` or rotation convention.

    Depending on the choice of convention for blockencoding, the same phase angles will produce different
    polynomial transformations. We provide the functionality to swap between blockencoding conventions and
    to transform the phase angles accordingly using the :code:`convention` keyword argument.

    .. seealso::

        :class:`~.QSVT` and `A Grand Unification of Quantum Algorithms <https://arxiv.org/pdf/2105.02859.pdf>`_.

    Args:
        A (tensor_like): the general :math:`(n \\times m)` matrix to be encoded
        angles (tensor_like): a list of angles by which to shift to obtain the desired polynomial
        wires (Iterable[int, str], Wires): the wires the template acts on
        convention (string): can be set to ``"Wx"`` to convert quantum signal processing angles in the
            `Wx` convention to QSVT angles.

    **Example**

    To implement QSVT in a circuit, we can use the following method:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> A = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> angles = np.array([0.1, 0.2, 0.3])
    >>> @qml.qnode(dev)
    ... def example_circuit(A):
    ...     qml.qsvt(A, angles, wires=[0, 1])
    ...     return qml.expval(qml.Z(0))

    The resulting circuit implements QSVT.

    >>> print(qml.draw(example_circuit)(A))
    0: ─╭QSVT─┤  <Z>
    1: ─╰QSVT─┤

    To see the implementation details, we can expand the circuit:

    >>> q_script = qml.tape.QuantumScript(ops=[qml.qsvt(A, angles, wires=[0, 1])])
    >>> print(q_script.expand().draw(decimals=2))
    0: ─╭∏_ϕ(0.30)─╭BlockEncode(M0)─╭∏_ϕ(0.20)─╭BlockEncode(M0)†─╭∏_ϕ(0.10)─┤
    1: ─╰∏_ϕ(0.30)─╰BlockEncode(M0)─╰∏_ϕ(0.20)─╰BlockEncode(M0)†─╰∏_ϕ(0.10)─┤
    """
    if qml.math.shape(A) == () or qml.math.shape(A) == (1,):
        A = qml.math.reshape(A, [1, 1])
    c, r = qml.math.shape(A)
    with qml.QueuingManager.stop_recording():
        UA = BlockEncode(A, wires=wires)
    projectors = []
    if convention == 'Wx':
        angles = _qsp_to_qsvt(angles)
        global_phase = (len(angles) - 1) % 4
        if global_phase:
            with qml.QueuingManager.stop_recording():
                global_phase_op = qml.GlobalPhase(-0.5 * np.pi * (4 - global_phase), wires=wires)
    for idx, phi in enumerate(angles):
        dim = c if idx % 2 else r
        with qml.QueuingManager.stop_recording():
            projectors.append(PCPhase(phi, dim=dim, wires=wires))
    projectors = projectors[::-1]
    if convention == 'Wx':
        return qml.prod(global_phase_op, QSVT(UA, projectors))
    return QSVT(UA, projectors)