from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_inverse_composite_standards():

    @cirq.value_equality
    class Gate(cirq.Gate):

        def __init__(self, param: 'cirq.TParamVal'):
            self._param = param

        def _decompose_(self, qubits):
            return cirq.S.on(qubits[0])

        def num_qubits(self) -> int:
            return 1

        def _has_unitary_(self):
            return True

        def _value_equality_values_(self):
            return (self._param,)

        def _parameter_names_(self) -> AbstractSet[str]:
            return cirq.parameter_names(self._param)

        def _is_parameterized_(self) -> bool:
            return cirq.is_parameterized(self._param)

        def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'Gate':
            return Gate(cirq.resolve_parameters(self._param, resolver, recursive))

        def __repr__(self):
            return f'C({self._param})'
    a = sympy.Symbol('a')
    g = cirq.inverse(Gate(a))
    assert cirq.is_parameterized(g)
    assert cirq.parameter_names(g) == {'a'}
    assert cirq.resolve_parameters(g, {a: 0}) == Gate(0) ** (-1)
    cirq.testing.assert_implements_consistent_protocols(g, global_vals={'C': Gate, 'a': a})