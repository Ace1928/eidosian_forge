from typing import Sequence, Callable
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires
from .optimization_utils import find_next_gate
@transform
def commute_controlled(tape: QuantumTape, direction='right') -> (Sequence[QuantumTape], Callable):
    """Quantum transform to move commuting gates past control and target qubits of controlled operations.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        direction (str): The direction in which to move single-qubit gates.
            Options are "right" (default), or "left". Single-qubit gates will
            be pushed through controlled operations as far as possible in the
            specified direction.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.


    **Example**

    >>> dev = qml.device('default.qubit', wires=3)

    You can apply the transform directly on :class:`QNode`:

    .. code-block:: python

        @partial(commute_controlled, direction="right")
        @qml.qnode(device=dev)
        def circuit(theta):
            qml.CZ(wires=[0, 2])
            qml.X(2)
            qml.S(wires=0)

            qml.CNOT(wires=[0, 1])

            qml.Y(1)
            qml.CRY(theta, wires=[0, 1])
            qml.PhaseShift(theta/2, wires=0)

            qml.Toffoli(wires=[0, 1, 2])
            qml.T(wires=0)
            qml.RZ(theta/2, wires=1)

            return qml.expval(qml.Z(0))

    >>> circuit(0.5)
    0.9999999999999999

    .. details::
        :title: Usage Details

        You can also apply it on quantum function.

        .. code-block:: python

            def qfunc(theta):
                qml.CZ(wires=[0, 2])
                qml.X(2)
                qml.S(wires=0)

                qml.CNOT(wires=[0, 1])

                qml.Y(1)
                qml.CRY(theta, wires=[0, 1])
                qml.PhaseShift(theta/2, wires=0)

                qml.Toffoli(wires=[0, 1, 2])
                qml.T(wires=0)
                qml.RZ(theta/2, wires=1)

                return qml.expval(qml.Z(0))

        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)(0.5))
        0: ─╭●──S─╭●────╭●─────────Rϕ(0.25)─╭●──T────────┤  <Z>
        1: ─│─────╰X──Y─╰RY(0.50)───────────├●──RZ(0.25)─┤
        2: ─╰Z──X───────────────────────────╰X───────────┤

        Diagonal gates on either side of control qubits do not affect the outcome
        of controlled gates; thus we can push all the single-qubit gates on the
        first qubit together on the right (and fuse them if desired). Similarly, X
        gates commute with the target of ``CNOT`` and ``Toffoli`` (and ``PauliY``
        with ``CRY``). We can use the transform to push single-qubit gates as
        far as possible through the controlled operations:

        >>> optimized_qfunc = commute_controlled(qfunc, direction="right")
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(0.5))
        0: ─╭●─╭●─╭●───────────╭●──S─────────Rϕ(0.25)──T─┤  <Z>
        1: ─│──╰X─╰RY(0.50)──Y─├●──RZ(0.25)──────────────┤
        2: ─╰Z─────────────────╰X──X─────────────────────┤

    """
    if direction not in ('left', 'right'):
        raise ValueError("Direction for commute_controlled must be 'left' or 'right'")
    if direction == 'right':
        op_list = _commute_controlled_right(tape.operations)
    else:
        op_list = _commute_controlled_left(tape.operations)
    new_tape = type(tape)(op_list, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]
    return ([new_tape], null_postprocessing)