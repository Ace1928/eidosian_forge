import cirq
from cirq.contrib.paulistring import (
def test_move_non_clifford_into_clifford():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2)
    c_left, c_right = convert_and_separate_circuit(c_orig)
    c_left_dag = pauli_string_dag_from_circuit(c_left)
    c_recombined1 = move_pauli_strings_into_circuit(c_left, c_right)
    c_recombined2 = move_pauli_strings_into_circuit(c_left_dag, c_right)
    _assert_no_multi_qubit_pauli_strings(c_recombined1)
    _assert_no_multi_qubit_pauli_strings(c_recombined2)
    gateset = cirq.CZTargetGateset()
    baseline_len = len(cirq.optimize_for_target_gateset(c_orig, gateset=gateset))
    opt_len1 = len(cirq.optimize_for_target_gateset(c_recombined1, gateset=gateset))
    opt_len2 = len(cirq.optimize_for_target_gateset(c_recombined2, gateset=gateset))
    assert opt_len1 <= baseline_len
    assert opt_len2 <= baseline_len