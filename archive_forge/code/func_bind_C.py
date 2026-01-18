from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Str
from sympy.core.sympify import sympify
from sympy.logic import true, false
from sympy.utilities.iterables import iterable
def bind_C(name=None):
    """ Creates an Attribute ``bind_C`` with a name.

    Parameters
    ==========

    name : str

    Examples
    ========

    >>> from sympy import fcode, Symbol
    >>> from sympy.codegen.ast import FunctionDefinition, real, Return
    >>> from sympy.codegen.fnodes import array, sum_, bind_C
    >>> a = Symbol('a', real=True)
    >>> s = Symbol('s', integer=True)
    >>> arr = array(a, dim=[s], intent='in')
    >>> body = [Return((sum_(a**2)/s)**.5)]
    >>> fd = FunctionDefinition(real, 'rms', [arr, s], body, attrs=[bind_C('rms')])
    >>> print(fcode(fd, source_format='free', standard=2003))
    real*8 function rms(a, s) bind(C, name="rms")
    real*8, dimension(s), intent(in) :: a
    integer*4 :: s
    rms = sqrt(sum(a**2)/s)
    end function

    """
    return Attribute('bind_C', [String(name)] if name else [])