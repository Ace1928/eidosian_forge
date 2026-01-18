import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_depolarizing_channel_invalid_probability():
    with pytest.raises(ValueError, match=re.escape('p(I) was greater than 1.')):
        cirq.depolarize(-0.1)
    with pytest.raises(ValueError, match=re.escape('p(I) was less than 0.')):
        cirq.depolarize(1.1)