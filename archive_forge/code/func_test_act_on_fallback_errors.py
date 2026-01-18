from typing import Any, Sequence, Tuple
from typing_extensions import Self
import numpy as np
import pytest
import cirq
def test_act_on_fallback_errors():
    state = ExampleSimulationState(fallback_result=False)
    with pytest.raises(ValueError, match='_act_on_fallback_ must return True or NotImplemented'):
        cirq.act_on(op, state)