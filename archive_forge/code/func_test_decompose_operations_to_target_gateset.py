import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers.optimize_for_target_gateset import _decompose_operations_to_target_gateset
import pytest
def test_decompose_operations_to_target_gateset():
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.T(q[0]), cirq.SWAP(*q), cirq.T(q[0]), cirq.SWAP(*q).with_tags('ignore'), cirq.measure(q[0], key='m'), cirq.X(q[1]).with_classical_controls('m'), cirq.Moment(cirq.T.on_each(*q)), cirq.SWAP(*q), cirq.T.on_each(*q))
    gateset = cirq.Gateset(cirq.H, cirq.CNOT)
    decomposer = lambda op, _: cirq.H(op.qubits[0]) if cirq.has_unitary(op) and cirq.num_qubits(op) == 1 else NotImplemented
    context = cirq.TransformerContext(tags_to_ignore=('ignore',))
    c_new = _decompose_operations_to_target_gateset(c_orig, gateset=gateset, decomposer=decomposer, context=context)
    cirq.testing.assert_has_diagram(c_new, "\n0: ───H───@───X───@───H───×['ignore']───M───────H───@───X───@───H───\n          │   │   │       │             ║           │   │   │\n1: ───────X───@───X───────×─────────────╫───X───H───X───@───X───H───\n                                        ║   ║\nm: ═════════════════════════════════════@═══^═══════════════════════")
    with pytest.raises(ValueError, match='Unable to convert'):
        _ = _decompose_operations_to_target_gateset(c_orig, gateset=gateset, decomposer=decomposer, context=context, ignore_failures=False)