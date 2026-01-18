from typing import Union, Tuple, cast
import numpy as np
import pytest
import sympy
import cirq
from cirq.type_workarounds import NotImplementedType
def test_extrapolatable_effect():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert cirq.ControlledGate(cirq.Z) ** 0.5 == cirq.ControlledGate(cirq.Z ** 0.5)
    assert cirq.ControlledGate(cirq.Z).on(a, b) ** 0.5 == cirq.ControlledGate(cirq.Z ** 0.5).on(a, b)
    assert cirq.ControlledGate(cirq.Z) ** 0.5 == cirq.ControlledGate(cirq.Z ** 0.5)