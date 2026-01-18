import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('op,max_two_cost', [(cirq.CCZ(*cirq.LineQubit.range(3)), 8), (cirq.CCX(*cirq.LineQubit.range(3)), 8), (cirq.CCZ(cirq.LineQubit(0), cirq.LineQubit(2), cirq.LineQubit(1)), 8), (cirq.CCZ(cirq.LineQubit(0), cirq.LineQubit(2), cirq.LineQubit(1)) ** sympy.Symbol('s'), 8), (cirq.CSWAP(*cirq.LineQubit.range(3)), 9), (cirq.CSWAP(*reversed(cirq.LineQubit.range(3))), 9), (cirq.CSWAP(cirq.LineQubit(1), cirq.LineQubit(0), cirq.LineQubit(2)), 12), (cirq.ThreeQubitDiagonalGate([2, 3, 5, 7, 11, 13, 17, 19])(cirq.LineQubit(1), cirq.LineQubit(2), cirq.LineQubit(3)), 8)])
def test_decomposition_cost(op: cirq.Operation, max_two_cost: int):
    ops = tuple(cirq.flatten_op_tree(cirq.decompose(op)))
    two_cost = len([e for e in ops if len(e.qubits) == 2])
    over_cost = len([e for e in ops if len(e.qubits) > 2])
    assert over_cost == 0
    assert two_cost == max_two_cost