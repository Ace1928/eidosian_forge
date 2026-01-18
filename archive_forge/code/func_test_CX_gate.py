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
def test_CX_gate():
    qasm = 'OPENQASM 2.0;\n     qreg q1[2];\n     qreg q2[2];\n     CX q1[0], q1[1];\n     CX q1, q2[0];\n     CX q2, q1;      \n'
    parser = QasmParser()
    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')
    q2_0 = cirq.NamedQubit('q2_0')
    q2_1 = cirq.NamedQubit('q2_1')
    expected_circuit = Circuit()
    expected_circuit.append(cirq.CNOT(q1_0, q1_1))
    expected_circuit.append(cirq.CNOT(q1_0, q2_0))
    expected_circuit.append(cirq.CNOT(q1_1, q2_0))
    expected_circuit.append(cirq.CNOT(q2_0, q1_0))
    expected_circuit.append(cirq.CNOT(q2_1, q1_1))
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.supportedFormat
    assert not parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2, 'q2': 2}