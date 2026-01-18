import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_custom_value_not_implemented():

    class BarImplicit:
        pass

    class BarExplicit:

        def _resolved_value_(self):
            return NotImplemented
    for cls in [BarImplicit, BarExplicit]:
        b = sympy.Symbol('b')
        bar = cls()
        r = cirq.ParamResolver({b: bar})
        assert r.value_of(b) == b