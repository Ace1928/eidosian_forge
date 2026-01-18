import itertools
import numpy as np
import pytest
import cirq
def test_gt_other_type():
    with pytest.raises(TypeError):
        _ = cirq.X > object()