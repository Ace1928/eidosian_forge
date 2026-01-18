from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import (Float, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing.repr import srepr
from sympy.abc import x
def test_dotprint():
    text = dotprint(x + 2, repeat=False)
    assert all((e in text for e in dotedges(x + 2, repeat=False)))
    assert all((n in text for n in [dotnode(expr, repeat=False) for expr in (x, Integer(2), x + 2)]))
    assert 'digraph' in text
    text = dotprint(x + x ** 2, repeat=False)
    assert all((e in text for e in dotedges(x + x ** 2, repeat=False)))
    assert all((n in text for n in [dotnode(expr, repeat=False) for expr in (x, Integer(2), x ** 2)]))
    assert 'digraph' in text
    text = dotprint(x + x ** 2, repeat=True)
    assert all((e in text for e in dotedges(x + x ** 2, repeat=True)))
    assert all((n in text for n in [dotnode(expr, pos=()) for expr in [x + x ** 2]]))
    text = dotprint(x ** x, repeat=True)
    assert all((e in text for e in dotedges(x ** x, repeat=True)))
    assert all((n in text for n in [dotnode(x, pos=(0,)), dotnode(x, pos=(1,))]))
    assert 'digraph' in text