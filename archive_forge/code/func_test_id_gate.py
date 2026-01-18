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
def test_id_gate():
    qasm = '\n     OPENQASM 2.0;\n     include "qelib1.inc";\n     qreg q[2];\n     id q;           \n'
    parser = QasmParser()
    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')
    expected_circuit = Circuit()
    expected_circuit.append(cirq.IdentityGate(num_qubits=1)(q0))
    expected_circuit.append(cirq.IdentityGate(num_qubits=1)(q1))
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}