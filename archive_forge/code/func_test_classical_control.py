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
def test_classical_control():
    qasm = 'OPENQASM 2.0;\n        qreg q[2];\n        creg a[1];\n        measure q[0] -> a[0];\n        if (a==1) CX q[0],q[1];\n    '
    parser = QasmParser()
    q_0 = cirq.NamedQubit('q_0')
    q_1 = cirq.NamedQubit('q_1')
    expected_circuit = cirq.Circuit(cirq.measure(q_0, key='a_0'), cirq.CNOT(q_0, q_1).with_classical_controls(sympy.Eq(sympy.Symbol('a_0'), 1)))
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.supportedFormat
    assert not parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}
    expected_generated_qasm = f'// Generated from Cirq v{cirq.__version__}\n\nOPENQASM 2.0;\ninclude "qelib1.inc";\n\n\n// Qubits: [q_0, q_1]\nqreg q[2];\ncreg m_a_0[1];\n\n\nmeasure q[0] -> m_a_0[0];\nif (m_a_0==1) cx q[0],q[1];\n'
    assert cirq.qasm(parsed_qasm.circuit) == expected_generated_qasm