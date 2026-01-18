from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import (Float, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing.repr import srepr
from sympy.abc import x
def test_dotedges():
    assert sorted(dotedges(x + 2, repeat=False)) == ['"Add(Integer(2), Symbol(\'x\'))" -> "Integer(2)";', '"Add(Integer(2), Symbol(\'x\'))" -> "Symbol(\'x\')";']
    assert sorted(dotedges(x + 2, repeat=True)) == ['"Add(Integer(2), Symbol(\'x\'))_()" -> "Integer(2)_(0,)";', '"Add(Integer(2), Symbol(\'x\'))_()" -> "Symbol(\'x\')_(1,)";']