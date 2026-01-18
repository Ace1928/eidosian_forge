from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_eq():
    d1 = ArithmeticDict(int, a=1, b=0)
    d2 = ArithmeticDict(int, a=1, b=1)
    d3 = ArithmeticDict(int, a=1)
    assert not d1 == d2
    assert d1 == d3