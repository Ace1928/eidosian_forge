import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
def test_cpmg_circuit():
    """Tests sub-component to make sure CPMG circuit is generated correctly."""
    q = cirq.GridQubit(1, 1)
    t = sympy.Symbol('t')
    circuit = t2._cpmg_circuit(q, t, 2)
    expected = cirq.Circuit(cirq.Y(q) ** 0.5, cirq.wait(q, nanos=t), cirq.X(q), cirq.wait(q, nanos=2 * t * sympy.Symbol('pulse_0')), cirq.X(q) ** sympy.Symbol('pulse_0'), cirq.wait(q, nanos=2 * t * sympy.Symbol('pulse_1')), cirq.X(q) ** sympy.Symbol('pulse_1'), cirq.wait(q, nanos=t))
    assert circuit == expected