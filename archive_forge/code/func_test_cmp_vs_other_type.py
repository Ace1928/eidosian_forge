from datetime import timedelta
import pytest
import cirq
from cirq import Timestamp, Duration
def test_cmp_vs_other_type():
    with pytest.raises(TypeError):
        _ = Timestamp() < Duration()
    with pytest.raises(TypeError):
        _ = Timestamp() < 0
    with pytest.raises(TypeError):
        _ = Timestamp() <= 0
    with pytest.raises(TypeError):
        _ = Timestamp() >= 0
    with pytest.raises(TypeError):
        _ = Timestamp() > 0