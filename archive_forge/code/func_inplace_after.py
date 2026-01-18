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
def inplace_after(self, ops: 'cirq.OP_TREE') -> 'cirq.MutablePauliString':
    """Propagates the pauli string from before to after a Clifford effect.

        If the old value of the MutablePauliString is $P$ and the Clifford
        operation is $C$, then the new value of the MutablePauliString is
        $C P C^\\dagger$.

        Args:
            ops: A stabilizer operation or nested collection of stabilizer
                operations.

        Returns:
            The mutable pauli string that was mutated.

        Raises:
            NotImplementedError: If any ops decompose into an unsupported
                Clifford gate.
        """
    for clifford in op_tree.flatten_to_ops(ops):
        for op in _decompose_into_cliffords(clifford):
            ps = [self.pauli_int_dict.pop(cast(TKey, q), 0) for q in op.qubits]
            if not any(ps):
                continue
            gate = op.gate
            if isinstance(gate, clifford_gate.SingleQubitCliffordGate):
                out = gate.pauli_tuple(_INT_TO_PAULI[ps[0] - 1])
                if out[1]:
                    self.coefficient *= -1
                self.pauli_int_dict[cast(TKey, op.qubits[0])] = PAULI_GATE_LIKE_TO_INDEX_MAP[out[0]]
            elif isinstance(gate, pauli_interaction_gate.PauliInteractionGate):
                q0, q1 = op.qubits
                p0 = _INT_TO_PAULI_OR_IDENTITY[ps[0]]
                p1 = _INT_TO_PAULI_OR_IDENTITY[ps[1]]
                kickback_0_to_1 = not protocols.commutes(p0, gate.pauli0)
                kickback_1_to_0 = not protocols.commutes(p1, gate.pauli1)
                kick0 = gate.pauli1 if kickback_0_to_1 else identity.I
                kick1 = gate.pauli0 if kickback_1_to_0 else identity.I
                self.__imul__({q0: p0, q1: kick0})
                self.__imul__({q0: kick1, q1: p1})
                if gate.invert0:
                    self.inplace_after(gate.pauli1(q1))
                if gate.invert1:
                    self.inplace_after(gate.pauli0(q0))
            else:
                raise NotImplementedError(f'Unrecognized decomposed Clifford: {op!r}')
    return self