from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_iadd():
    d1 = ArithmeticDict(int, [('a', 1), ('b', 2)])
    d1 += 3
    assert d1['a'] == 4
    assert d1['b'] == 5
    assert d1['c'] == 0