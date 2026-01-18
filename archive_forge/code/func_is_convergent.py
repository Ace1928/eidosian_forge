from typing import Tuple as tTuple
from .expr_with_intlimits import ExprWithIntLimits
from .summations import Sum, summation, _dummy_with_inherited_properties_concrete
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import Derivative
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import RisingFactorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.polys import quo, roots
def is_convergent(self):
    """
        See docs of :obj:`.Sum.is_convergent()` for explanation of convergence
        in SymPy.

        Explanation
        ===========

        The infinite product:

        .. math::

            \\prod_{1 \\leq i < \\infty} f(i)

        is defined by the sequence of partial products:

        .. math::

            \\prod_{i=1}^{n} f(i) = f(1) f(2) \\cdots f(n)

        as n increases without bound. The product converges to a non-zero
        value if and only if the sum:

        .. math::

            \\sum_{1 \\leq i < \\infty} \\log{f(n)}

        converges.

        Examples
        ========

        >>> from sympy import Product, Symbol, cos, pi, exp, oo
        >>> n = Symbol('n', integer=True)
        >>> Product(n/(n + 1), (n, 1, oo)).is_convergent()
        False
        >>> Product(1/n**2, (n, 1, oo)).is_convergent()
        False
        >>> Product(cos(pi/n), (n, 1, oo)).is_convergent()
        True
        >>> Product(exp(-n**2), (n, 1, oo)).is_convergent()
        False

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Infinite_product
        """
    sequence_term = self.function
    log_sum = log(sequence_term)
    lim = self.limits
    try:
        is_conv = Sum(log_sum, *lim).is_convergent()
    except NotImplementedError:
        if Sum(sequence_term - 1, *lim).is_absolutely_convergent() is S.true:
            return S.true
        raise NotImplementedError('The algorithm to find the product convergence of %s is not yet implemented' % sequence_term)
    return is_conv