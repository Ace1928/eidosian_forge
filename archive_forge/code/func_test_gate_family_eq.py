from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
def test_gate_family_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.GateFamily(CustomX))
    eq.add_equality_group(cirq.GateFamily(CustomX ** 3))
    eq.add_equality_group(cirq.GateFamily(CustomX, name='custom_name', description='custom_description'), cirq.GateFamily(CustomX ** 3, name='custom_name', description='custom_description'))
    eq.add_equality_group(cirq.GateFamily(CustomXPowGate))
    eq.add_equality_group(cirq.GateFamily(CustomXPowGate, name='custom_name', description='custom_description'))