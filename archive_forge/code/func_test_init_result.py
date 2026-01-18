import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
def test_init_result():
    data = pd.DataFrame(columns=['delay_ns', 'false_count', 'true_count'], index=range(2), data=[[100.0, 0, 10], [1000.0, 10, 0]])
    result = cirq.experiments.T1DecayResult(data)
    assert result.data is data