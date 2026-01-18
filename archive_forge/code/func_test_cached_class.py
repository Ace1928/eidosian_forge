from monty.design_patterns import cached_class, singleton
def test_cached_class(self):
    a1a = A(1)
    a1b = A(1)
    a2 = A(2)
    assert id(a1a) == id(a1b)
    assert id(a1a) != id(a2)