import numpy as np
import cirq
def test_equal_up_to_global_numeric_iterables():
    assert cirq.equal_up_to_global_phase([], [], atol=1e-09)
    assert cirq.equal_up_to_global_phase([[]], [[]], atol=1e-09)
    assert cirq.equal_up_to_global_phase([1j, 1], [1j, 1], atol=1e-09)
    assert cirq.equal_up_to_global_phase([1j, 1j], [1 + 0.1j, 1 + 0.1j], atol=0.01)
    assert not cirq.equal_up_to_global_phase([1j, 1j], [1 + 0.1j, 1 - 0.1j], atol=0.01)
    assert not cirq.equal_up_to_global_phase([1j, 1j], [1 + 0.1j, 1 + 0.1j], atol=0.001)
    assert not cirq.equal_up_to_global_phase([1j, -1j], [1, 1], atol=0.0)
    assert not cirq.equal_up_to_global_phase([1j, 1], [1, 1j], atol=0.0)
    assert not cirq.equal_up_to_global_phase([1j, 1], [1j, 1, 0], atol=0.0)
    assert cirq.equal_up_to_global_phase((1j, 1j), (1, 1 + 0.0001), atol=0.001)
    assert not cirq.equal_up_to_global_phase((1j, 1j), (1, 1 + 0.0001), atol=1e-05)
    assert not cirq.equal_up_to_global_phase((1j, 1), (1, 1j), atol=1e-09)