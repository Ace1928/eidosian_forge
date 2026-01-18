from typing import List
import cirq
def test_merge_single_qubit_gates_to_phxz_deep():
    a = cirq.NamedQubit('a')
    c_nested = cirq.FrozenCircuit(cirq.H(a), cirq.Z(a), cirq.H(a).with_tags('ignore'))
    c_nested_merged = cirq.FrozenCircuit(_phxz(-0.5, 0.5, 0).on(a), cirq.H(a).with_tags('ignore'))
    c_orig = cirq.Circuit(c_nested, cirq.CircuitOperation(c_nested).repeat(4).with_tags('ignore'), c_nested, cirq.CircuitOperation(c_nested).repeat(5).with_tags('preserve_tags'), c_nested, cirq.CircuitOperation(c_nested).repeat(6))
    c_expected = cirq.Circuit(c_nested_merged, cirq.CircuitOperation(c_nested).repeat(4).with_tags('ignore'), c_nested_merged, cirq.CircuitOperation(c_nested_merged).repeat(5).with_tags('preserve_tags'), c_nested_merged, cirq.CircuitOperation(c_nested_merged).repeat(6))
    context = cirq.TransformerContext(tags_to_ignore=['ignore'], deep=True)
    c_new = cirq.merge_single_qubit_gates_to_phxz(c_orig, context=context)
    cirq.testing.assert_same_circuits(c_new, c_expected)