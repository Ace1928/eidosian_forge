import mpmath as mp
from collections.abc import Callable
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import diff
from sympy.core.expr import Expr
from sympy.core.kind import _NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, uniquely_named_symbol
from sympy.core.sympify import sympify, _sympify
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy.polys import cancel
from sympy.printing import sstr
from sympy.printing.defaults import Printable
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import flatten, NotIterable, is_sequence, reshape
from sympy.utilities.misc import as_int, filldedent
from .common import (
from .utilities import _iszero, _is_zero_after_expand_mul, _simplify
from .determinant import (
from .reductions import _is_echelon, _echelon_form, _rank, _rref
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize
from .eigen import (
from .decompositions import (
from .graph import (
from .solvers import (
from .inverse import (
def analytic_func(self, f, x):
    """
        Computes f(A) where A is a Square Matrix
        and f is an analytic function.

        Examples
        ========

        >>> from sympy import Symbol, Matrix, S, log

        >>> x = Symbol('x')
        >>> m = Matrix([[S(5)/4, S(3)/4], [S(3)/4, S(5)/4]])
        >>> f = log(x)
        >>> m.analytic_func(f, x)
        Matrix([
        [     0, log(2)],
        [log(2),      0]])

        Parameters
        ==========

        f : Expr
            Analytic Function
        x : Symbol
            parameter of f

        """
    f, x = (_sympify(f), _sympify(x))
    if not self.is_square:
        raise NonSquareMatrixError
    if not x.is_symbol:
        raise ValueError('{} must be a symbol.'.format(x))
    if x not in f.free_symbols:
        raise ValueError('{} must be a parameter of {}.'.format(x, f))
    if x in self.free_symbols:
        raise ValueError('{} must not be a parameter of {}.'.format(x, self))
    eigen = self.eigenvals()
    max_mul = max(eigen.values())
    derivative = {}
    dd = f
    for i in range(max_mul - 1):
        dd = diff(dd, x)
        derivative[i + 1] = dd
    n = self.shape[0]
    r = self.zeros(n)
    f_val = self.zeros(n, 1)
    row = 0
    for i in eigen:
        mul = eigen[i]
        f_val[row] = f.subs(x, i)
        if f_val[row].is_number and (not f_val[row].is_complex):
            raise ValueError('Cannot evaluate the function because the function {} is not analytic at the given eigenvalue {}'.format(f, f_val[row]))
        val = 1
        for a in range(n):
            r[row, a] = val
            val *= i
        if mul > 1:
            coe = [1 for ii in range(n)]
            deri = 1
            while mul > 1:
                row = row + 1
                mul -= 1
                d_i = derivative[deri].subs(x, i)
                if d_i.is_number and (not d_i.is_complex):
                    raise ValueError('Cannot evaluate the function because the derivative {} is not analytic at the given eigenvalue {}'.format(derivative[deri], d_i))
                f_val[row] = d_i
                for a in range(n):
                    if a - deri + 1 <= 0:
                        r[row, a] = 0
                        coe[a] = 0
                        continue
                    coe[a] = coe[a] * (a - deri + 1)
                    r[row, a] = coe[a] * pow(i, a - deri)
                deri += 1
        row += 1
    c = r.solve(f_val)
    ans = self.zeros(n)
    pre = self.eye(n)
    for i in range(n):
        ans = ans + c[i] * pre
        pre *= self
    return ans