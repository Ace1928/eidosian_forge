import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
def test_controlled_mixture():
    a, b = cirq.LineQubit.range(2)
    c_yes = cirq.ControlledOperation(controls=[b], sub_operation=cirq.phase_flip(0.25).on(a))
    assert cirq.has_mixture(c_yes)
    assert cirq.approx_eq(cirq.mixture(c_yes), [(0.75, np.eye(4)), (0.25, cirq.unitary(cirq.CZ))])