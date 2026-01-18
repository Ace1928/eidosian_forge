import cirq
from cirq.interop.quirk.cells.qubit_permutation_cells import QuirkQubitPermutationGate
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_left_rotate():
    assert_url_to_circuit_returns('{"cols":[["<<4"]]}', maps={0: 0, 1: 8, 2: 1, 4: 2, 8: 4, 15: 15, 10: 5})