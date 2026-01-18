import numpy as np
import pytest
import cirq
def test_custom_class_repr():

    class CustomRepr:
        setup_code = 'class CustomRepr:\n            def __init__(self, eq_val):\n                self.eq_val = eq_val\n            def __pow__(self, exponent):\n                return self\n        '

        def __init__(self, eq_val, repr_str: str):
            self.eq_val = eq_val
            self.repr_str = repr_str

        def __eq__(self, other):
            return self.eq_val == getattr(other, 'eq_val', None)

        def __ne__(self, other):
            return not self == other

        def __repr__(self):
            return self.repr_str
    cirq.testing.assert_equivalent_repr(CustomRepr('b', "CustomRepr('b')"), setup_code=CustomRepr.setup_code)
    cirq.testing.assert_equivalent_repr(CustomRepr('a', "CustomRepr('a')"), setup_code=CustomRepr.setup_code)
    with pytest.raises(AssertionError, match='eval\\(repr\\(value\\)\\): a'):
        cirq.testing.assert_equivalent_repr(CustomRepr('a', "'a'"))
    with pytest.raises(AssertionError, match='eval\\(repr\\(value\\)\\): 1'):
        cirq.testing.assert_equivalent_repr(CustomRepr('a', '1'))
    with pytest.raises(AssertionError, match='eval\\(repr\\(value\\)\\): a'):
        cirq.testing.assert_equivalent_repr(CustomRepr('a', "'a'"))
    with pytest.raises(AssertionError, match='SyntaxError'):
        cirq.testing.assert_equivalent_repr(CustomRepr('a', '('))
    with pytest.raises(AssertionError, match='SyntaxError'):
        cirq.testing.assert_equivalent_repr(CustomRepr('a', 'return 1'))
    with pytest.raises(AssertionError, match='dottable'):
        cirq.testing.assert_equivalent_repr(CustomRepr(5, 'CustomRepr(5)**1'), setup_code=CustomRepr.setup_code)