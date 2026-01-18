import functools
import itertools
import math
import random
import numpy as np
import pytest
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.ops.boolean_hamiltonian as bh
def test_gate_with_custom_names():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    gate = cirq.BooleanHamiltonianGate(['a', 'b'], ['a'], 0.1)
    assert cirq.decompose(gate.on(q0, q1)) == [cirq.Rz(rads=-0.05).on(q0)]
    assert cirq.decompose_once_with_qubits(gate, (q0, q1)) == [cirq.Rz(rads=-0.05).on(q0)]
    assert cirq.decompose(gate.on(q2, q3)) == [cirq.Rz(rads=-0.05).on(q2)]
    assert cirq.decompose_once_with_qubits(gate, (q2, q3)) == [cirq.Rz(rads=-0.05).on(q2)]
    with pytest.raises(ValueError, match='Wrong number of qubits'):
        gate.on(q2)
    with pytest.raises(ValueError, match='Wrong shape of qids'):
        gate.on(q0, cirq.LineQid(1, 3))