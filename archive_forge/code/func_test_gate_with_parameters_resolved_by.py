import itertools
import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_gate_with_parameters_resolved_by(resolve_fn):
    gate = cirq.PauliStringPhasorGate(dps_empty, exponent_neg=sympy.Symbol('a'))
    resolver = cirq.ParamResolver({'a': 0.1})
    actual = resolve_fn(gate, resolver)
    expected = cirq.PauliStringPhasorGate(dps_empty, exponent_neg=0.1)
    assert actual == expected