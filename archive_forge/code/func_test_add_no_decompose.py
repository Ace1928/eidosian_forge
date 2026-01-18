import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('a,b', itertools.product(range(2 ** 3), repeat=2))
@allow_deprecated_cirq_ft_use_in_tests
def test_add_no_decompose(a, b):
    num_bits = 5
    qubits = cirq.LineQubit.range(2 * num_bits)
    op = cirq_ft.AdditionGate(num_bits).on(*qubits)
    circuit = cirq.Circuit(op)
    basis_map = {}
    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')
    out_bin = format(a + b, f'0{num_bits}b')
    true_out_int = a + b
    input_int = int(a_bin + b_bin, 2)
    output_int = int(a_bin + out_bin, 2)
    assert true_out_int == int(out_bin, 2)
    basis_map[input_int] = output_int
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)