import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def conjugated_by(self, clifford: 'cirq.OP_TREE') -> 'PauliString':
    """Returns the Pauli string conjugated by a clifford operation.

        The product-of-Paulis $P$ conjugated by the Clifford operation $C$ is

            $$
            C^\\dagger P C
            $$

        For example, conjugating a +Y operation by an S operation results in a
        +X operation (as opposed to a -X operation).

        In a circuit diagram where `P` is a pauli string observable immediately
        after a Clifford operation `C`, the pauli string `P.conjugated_by(C)` is
        the equivalent pauli string observable just before `C`.

            --------------------------C---P---

            = ---C---P------------------------

            = ---C---P---------C^-1---C-------

            = ---C---P---C^-1---------C-------

            = --(C^-1 · P · C)--------C-------

            = ---P.conjugated_by(C)---C-------

        Analogously, a Pauli product P can be moved from before a Clifford C in
        a circuit diagram to after the Clifford C by conjugating P by the
        inverse of C:

            ---P---C---------------------------

            = -----C---P.conjugated_by(C^-1)---

        Args:
            clifford: The Clifford operation to conjugate by. This can be an
                individual operation, or a tree of operations.

                Note that the composite Clifford operation defined by a sequence
                of operations is equivalent to a circuit containing those
                operations in the given order. Somewhat counter-intuitively,
                this means that the operations in the sequence are conjugated
                onto the Pauli string in reverse order. For example,
                `P.conjugated_by([C1, C2])` is equivalent to
                `P.conjugated_by(C2).conjugated_by(C1)`.

        Examples:
            >>> a, b = cirq.LineQubit.range(2)
            >>> print(cirq.X(a).conjugated_by(cirq.CZ(a, b)))
            X(q(0))*Z(q(1))
            >>> print(cirq.X(a).conjugated_by(cirq.S(a)))
            -Y(q(0))
            >>> print(cirq.X(a).conjugated_by([cirq.H(a), cirq.CNOT(a, b)]))
            Z(q(0))*X(q(1))

        Returns:
            The Pauli string conjugated by the given Clifford operation.
        """
    pauli_map = dict(self._qubit_pauli_map)
    should_negate = False
    for op in list(op_tree.flatten_to_ops(clifford))[::-1]:
        if pauli_map.keys().isdisjoint(set(op.qubits)):
            continue
        for clifford_op in _decompose_into_cliffords(op)[::-1]:
            if pauli_map.keys().isdisjoint(set(clifford_op.qubits)):
                continue
            should_negate ^= _pass_operation_over(pauli_map, clifford_op, False)
    coef = -self._coefficient if should_negate else self.coefficient
    return PauliString(qubit_pauli_map=pauli_map, coefficient=coef)