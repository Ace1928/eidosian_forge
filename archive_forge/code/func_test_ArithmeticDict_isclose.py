from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_isclose():
    d1 = ArithmeticDict(float)
    d2 = ArithmeticDict(float)
    assert d1.isclose(d2)
    d1['a'] = 2
    assert not d1.isclose(d2)
    d2['a'] = 2 + 1e-15
    assert d1.isclose(d2)
    d2['b'] = 1e-15
    assert not d1.isclose(d2)
    assert d1.isclose(d2, atol=1e-14)