from typing import Any, Sequence, Tuple
from typing_extensions import Self
import numpy as np
import pytest
import cirq
def test_act_on_fallback_fails():
    state = ExampleSimulationState(fallback_result=NotImplemented)
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(op, state)