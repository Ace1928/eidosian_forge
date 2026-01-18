import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_multi_in_less_equal_than_gate():
    qubits = cirq.LineQubit.range(7)
    op = cirq_ft.LessThanEqualGate(3, 3).on(*qubits)
    circuit = cirq.Circuit(op)
    basis_map = {}
    for in1, in2 in itertools.product(range(2 ** 3), repeat=2):
        for target_reg_val in range(2):
            target_bin = bin(target_reg_val)[2:]
            in1_bin = format(in1, '03b')
            in2_bin = format(in2, '03b')
            out_bin = bin(target_reg_val ^ (in1 <= in2))[2:]
            true_out_int = target_reg_val ^ (in1 <= in2)
            input_int = int(in1_bin + in2_bin + target_bin, 2)
            output_int = int(in1_bin + in2_bin + out_bin, 2)
            assert true_out_int == int(out_bin, 2)
            basis_map[input_int] = output_int
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    circuit += op ** (-1)
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(len(qubits)), circuit)