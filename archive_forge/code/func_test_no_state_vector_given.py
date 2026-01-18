import math
import cirq
import pytest
import numpy as np
import cirq_web
def test_no_state_vector_given():
    with pytest.raises(ValueError):
        cirq_web.BlochSphere()