from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_imul():
    d1 = ArithmeticDict(int, [('a', 1), ('b', 2)])
    d1 *= 3
    assert d1['a'] == 3
    assert d1['b'] == 6
    assert d1['c'] == 0