import numpy as np
import pytest
import cirq
import cirq.testing
def test_dirac_notation_precision():
    sqrt = np.sqrt(0.5)
    assert_dirac_notation_numpy([sqrt, sqrt], '0.7|0⟩ + 0.7|1⟩', decimals=1)
    assert_dirac_notation_python([sqrt, sqrt], '0.707|0⟩ + 0.707|1⟩', decimals=3)