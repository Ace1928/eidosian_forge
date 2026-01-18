import pytest
import cirq
from cirq import circuits
def test_incorrect_mappings():
    a, b, c = cirq.LineQubit.range(3)
    with pytest.raises(AssertionError, match='0b001 \\(1\\) was mapped to 0b100 \\(4\\) instead of 0b010 \\(2\\)'):
        cirq.testing.assert_equivalent_computational_basis_map(maps={1: 2, 2: 4, 4: 1}, circuit=circuits.Circuit(cirq.SWAP(a, c), cirq.I(b)))