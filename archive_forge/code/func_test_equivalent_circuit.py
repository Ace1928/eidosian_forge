import pytest
import numpy as np
import sympy
import cirq
def test_equivalent_circuit():
    qreg = cirq.LineQubit.range(4)
    oldc = cirq.Circuit()
    newc = cirq.Circuit()
    single_qubit_gates = [cirq.X ** (1 / 2), cirq.Y ** (1 / 3), cirq.Z ** (-1)]
    for gate in single_qubit_gates:
        for qubit in qreg:
            oldc.append(gate.on(qubit))
        newc.append(cirq.ParallelGate(gate, 4)(*qreg))
    cirq.testing.assert_has_diagram(newc, oldc.to_text_diagram())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(oldc, newc, atol=1e-06)