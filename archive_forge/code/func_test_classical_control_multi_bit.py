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
def test_classical_control_multi_bit():
    qasm = 'OPENQASM 2.0;\n        qreg q[2];\n        creg a[2];\n        measure q[0] -> a[0];\n        measure q[0] -> a[1];\n        if (a==1) CX q[0],q[1];\n    '
    parser = QasmParser()
    q_0 = cirq.NamedQubit('q_0')
    q_1 = cirq.NamedQubit('q_1')
    expected_circuit = cirq.Circuit(cirq.measure(q_0, key='a_0'), cirq.measure(q_0, key='a_1'), cirq.CNOT(q_0, q_1).with_classical_controls(sympy.Eq(sympy.Symbol('a_0'), 1), sympy.Eq(sympy.Symbol('a_1'), 0)))
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.supportedFormat
    assert not parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}
    with pytest.raises(ValueError, match='QASM does not support multiple conditions'):
        _ = cirq.qasm(parsed_qasm.circuit)