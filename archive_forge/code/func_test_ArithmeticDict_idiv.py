from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_idiv():
    d1 = ArithmeticDict(int, [('a', 6), ('b', 9)])
    d1 /= 3
    assert d1['a'] == 2
    assert d1['b'] == 3
    assert d1['c'] == 0