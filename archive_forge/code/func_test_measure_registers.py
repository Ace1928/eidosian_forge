from typing import Callable
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing as ct
from cirq import Circuit
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._parser import QasmParser
def test_measure_registers():
    qasm = 'OPENQASM 2.0;\n         include "qelib1.inc";\n         qreg q1[3];\n         creg c1[3];                        \n         measure q1 -> c1;       \n    '
    parser = QasmParser()
    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')
    q1_2 = cirq.NamedQubit('q1_2')
    expected_circuit = Circuit()
    expected_circuit.append(cirq.MeasurementGate(num_qubits=1, key='c1_0').on(q1_0))
    expected_circuit.append(cirq.MeasurementGate(num_qubits=1, key='c1_1').on(q1_1))
    expected_circuit.append(cirq.MeasurementGate(num_qubits=1, key='c1_2').on(q1_2))
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 3}
    assert parsed_qasm.cregs == {'c1': 3}