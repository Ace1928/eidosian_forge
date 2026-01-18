from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
def test_gate_family_immutable():
    g = cirq.GateFamily(CustomX)
    with pytest.raises(AttributeError, match="(can't set attribute)|(property 'gate' of 'GateFamily' object has no setter)"):
        g.gate = CustomXPowGate
    with pytest.raises(AttributeError, match="(can't set attribute)|(property 'name' of 'GateFamily' object has no setter)"):
        g.name = 'new name'
    with pytest.raises(AttributeError, match="(can't set attribute)|(property 'description' of 'GateFamily' object has no setter)"):
        g.description = 'new description'