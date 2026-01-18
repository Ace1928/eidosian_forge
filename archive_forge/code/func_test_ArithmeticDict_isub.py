from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_isub():
    d1 = ArithmeticDict(int, [('a', 1), ('b', 2)])
    d1 -= 7
    assert d1['a'] == -6
    assert d1['b'] == -5
    assert d1['c'] == 0