import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_custom_resolved_value():

    class Foo:

        def _resolved_value_(self):
            return self

    class Baz:

        def _resolved_value_(self):
            return 'Baz'
    foo = Foo()
    baz = Baz()
    a = sympy.Symbol('a')
    b = sympy.Symbol('c')
    r = cirq.ParamResolver({a: foo, b: baz})
    assert r.value_of(a) is foo
    assert r.value_of(b) == 'Baz'