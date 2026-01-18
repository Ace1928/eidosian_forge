import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers.optimize_for_target_gateset import _decompose_operations_to_target_gateset
import pytest
def test_decompose_operations_raises_on_stuck():
    c_orig = cirq.Circuit(cirq.X(cirq.NamedQubit('q')).with_tags('ignore'))
    gateset = cirq.Gateset(cirq.Y)
    with pytest.raises(ValueError, match='Unable to convert'):
        _ = _decompose_operations_to_target_gateset(c_orig, gateset=gateset, ignore_failures=False)
    c_new = _decompose_operations_to_target_gateset(c_orig, context=cirq.TransformerContext(tags_to_ignore=('ignore',)), gateset=gateset, ignore_failures=False)
    cirq.testing.assert_same_circuits(c_orig, c_new)