from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
from sympy.testing.pytest import raises, warns
def test_halt_method_resolution():
    g = [0]

    def on_ambiguity(a, b):
        g[0] += 1
    f = Dispatcher('f')
    halt_ordering()

    def func(*args):
        pass
    f.add((int, object), func)
    f.add((object, int), func)
    assert g == [0]
    restart_ordering(on_ambiguity=on_ambiguity)
    assert g == [1]
    assert set(f.ordering) == {(int, object), (object, int)}