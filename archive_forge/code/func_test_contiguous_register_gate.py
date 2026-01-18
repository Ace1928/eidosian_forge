import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_contiguous_register_gate():
    gate = cirq_ft.ContiguousRegisterGate(3, 6)
    circuit = cirq.Circuit(gate.on(*cirq.LineQubit.range(12)))
    basis_map = {}
    for p in range(2 ** 3):
        for q in range(p):
            inp = f'0b_{p:03b}_{q:03b}_{0:06b}'
            out = f'0b_{p:03b}_{q:03b}_{p * (p - 1) // 2 + q:06b}'
            basis_map[int(inp, 2)] = int(out, 2)
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq_ft')
    gate = cirq_ft.ContiguousRegisterGate(2, 4)
    assert gate ** (-1) is gate
    u = cirq.unitary(gate)
    np.testing.assert_allclose(u @ u, np.eye(2 ** cirq.num_qubits(gate)))
    expected_wire_symbols = ('In(x)',) * 2 + ('In(y)',) * 2 + ('+(x(x-1)/2 + y)',) * 4
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_wire_symbols
    assert gate.with_registers([2] * 3, [2] * 3, [2] * 6) == cirq_ft.ContiguousRegisterGate(3, 6)