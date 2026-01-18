from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_floordiv():
    d1 = ArithmeticDict(int, [('a', 6), ('b', 9)])
    d2 = d1 // 2
    assert d2['a'] == 3
    assert d2['b'] == 4
    d2 = d1.copy() // 5
    assert d2['a'] == 1
    assert d2['b'] == 1
    d2 = 55 // d1
    assert d2['a'] == 9
    assert d2['b'] == 6
    assert d2['c'] == 0
    d1['c'] = 1
    d3 = (6 * d2 + 1) // d1
    assert d3['a'] == 9
    assert d3['b'] == 4
    assert d3['c'] == 1