from dill.source import getsource, getname, _wrap, likely_import
from dill.source import getimportable
from dill._dill import IS_PYPY
import sys
def test_getsource():
    assert getsource(f) == 'f = lambda x: x**2\n'
    assert getsource(g) == 'def g(x): return f(x) - x\n'
    assert getsource(h) == 'def h(x):\n  def g(x): return x\n  return g(x) - x\n'
    assert getname(f) == 'f'
    assert getname(g) == 'g'
    assert getname(h) == 'h'
    assert _wrap(f)(4) == 16
    assert _wrap(g)(4) == 12
    assert _wrap(h)(4) == 0
    assert getname(Foo) == 'Foo'
    assert getname(Bar) == 'Bar'
    assert getsource(Bar) == 'class Bar:\n  pass\n'
    assert getsource(Foo) == 'class Foo(object):\n  def bar(self, x):\n    return x*x+x\n'