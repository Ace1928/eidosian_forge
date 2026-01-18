from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
def test_invalid_gate_family():
    with pytest.raises(ValueError, match='instance or subclass of `cirq.Gate`'):
        _ = cirq.GateFamily(gate=cirq.Operation)
    with pytest.raises(ValueError, match='non-parameterized instance of `cirq.Gate`'):
        _ = cirq.GateFamily(gate=CustomX ** sympy.Symbol('theta'))
    with pytest.raises(ValueError, match='cannot be in both'):
        _ = cirq.GateFamily(gate=cirq.H, tags_to_accept={'a', 'b'}, tags_to_ignore={'b', 'c'})