from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_all_non_negative():
    d1 = ArithmeticDict(float)
    assert d1.all_non_negative()
    d1['a'] = 0.1
    assert d1.all_non_negative()
    d1['b'] = 0
    assert d1.all_non_negative()
    d1['b'] -= 1e-15
    assert not d1.all_non_negative()