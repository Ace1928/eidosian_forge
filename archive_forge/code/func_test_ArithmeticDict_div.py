from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_div():
    d1 = ArithmeticDict(int, [('a', 6), ('b', 9)])
    d2 = d1 / 3
    assert d2['a'] == 2
    assert d2['b'] == 3
    d2 = d1.copy() / 3
    assert d2['a'] == 2
    assert d2['b'] == 3
    d2 = 54 / d1
    assert d2['a'] == 9
    assert d2['b'] == 6
    assert d2['c'] == 0
    d1['c'] = 1
    d3 = 6 * d2 / d1
    assert d3['a'] == 9
    assert d3['b'] == 4
    assert d3['c'] == 0