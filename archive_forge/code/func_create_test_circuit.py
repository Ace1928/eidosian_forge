from typing import List, Sequence, Tuple
import numpy as np
import sympy
import cirq
from cirq.contrib.custom_simulators.custom_state_simulator import CustomStateSimulator
def create_test_circuit():
    q0, q1 = cirq.LineQid.range(2, dimension=3)
    x = cirq.XPowGate(dimension=3)
    return cirq.Circuit(x(q0), cirq.measure(q0, key='a'), x(q0).with_classical_controls('a'), cirq.CircuitOperation(cirq.FrozenCircuit(x(q1), cirq.measure(q1, key='b')), repeat_until=cirq.SympyCondition(sympy.Eq(sympy.Symbol('b'), 2)), use_repetition_ids=False))