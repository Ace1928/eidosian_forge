import pytest
import numpy as np
import cirq
def test_missing_mixture():
    with pytest.raises(TypeError, match='_mixture_'):
        cirq.validate_mixture(NoMethod)