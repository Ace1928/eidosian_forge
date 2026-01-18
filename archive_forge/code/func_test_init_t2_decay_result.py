import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
def test_init_t2_decay_result():
    x_data = pd.DataFrame(columns=['delay_ns', 0, 1], index=range(2), data=[[100.0, 0, 10], [1000.0, 10, 0]])
    y_data = pd.DataFrame(columns=['delay_ns', 0, 1], index=range(2), data=[[100.0, 5, 5], [1000.0, 5, 5]])
    result = cirq.experiments.T2DecayResult(x_data, y_data)
    assert result
    bad_data = pd.DataFrame(columns=['delay_ms', 0, 1], index=range(2), data=[[100.0, 0, 10], [1000.0, 10, 0]])
    with pytest.raises(ValueError):
        cirq.experiments.T2DecayResult(bad_data, y_data)
    with pytest.raises(ValueError):
        cirq.experiments.T2DecayResult(x_data, bad_data)