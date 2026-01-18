from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
@pytest.mark.parametrize('gate', [cirq.X, cirq.XPowGate(), cirq.XPowGate])
@pytest.mark.parametrize('name, description', [(None, None), ('custom_name', 'custom_description')])
def test_gate_family_json(gate, name, description):
    g = cirq.GateFamily(gate, name=name, description=description)
    g_json = cirq.to_json(g)
    assert cirq.read_json(json_text=g_json) == g