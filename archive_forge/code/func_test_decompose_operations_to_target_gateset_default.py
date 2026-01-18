import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers.optimize_for_target_gateset import _decompose_operations_to_target_gateset
import pytest
def test_decompose_operations_to_target_gateset_default():
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.T(q[0]), cirq.SWAP(*q), cirq.T(q[0]), cirq.SWAP(*q).with_tags('ignore'), cirq.measure(q[0], key='m'), cirq.X(q[1]).with_classical_controls('m'), cirq.Moment(cirq.T.on_each(*q)), cirq.SWAP(*q), cirq.T.on_each(*q))
    cirq.testing.assert_has_diagram(c_orig, "\n0: ───T───×───T───×['ignore']───M───────T───×───T───\n          │       │             ║           │\n1: ───────×───────×─────────────╫───X───T───×───T───\n                                ║   ║\nm: ═════════════════════════════@═══^═══════════════")
    context = cirq.TransformerContext(tags_to_ignore=('ignore',))
    c_new = _decompose_operations_to_target_gateset(c_orig, context=context)
    cirq.testing.assert_has_diagram(c_new, "\n0: ───T────────────@───Y^-0.5───@───Y^0.5────@───────────T───×['ignore']───M───────T────────────@───Y^-0.5───@───Y^0.5────@───────────T───\n                   │            │            │               │             ║                    │            │            │\n1: ───────Y^-0.5───@───Y^0.5────@───Y^-0.5───@───Y^0.5───────×─────────────╫───X───T───Y^-0.5───@───Y^0.5────@───Y^-0.5───@───Y^0.5───T───\n                                                                           ║   ║\nm: ════════════════════════════════════════════════════════════════════════@═══^══════════════════════════════════════════════════════════\n")