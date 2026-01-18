import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_short_circuits_act_on():
    args = mock.Mock(cirq.SimulationState)
    args._act_on_fallback_.side_effect = mock.Mock(side_effect=Exception('No!'))
    cirq.act_on(cirq.IdentityGate(1)(cirq.LineQubit(0)), args)