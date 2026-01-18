from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j
from sympy.printing.precedence import PRECEDENCE
def cg_simp(e):
    """Simplify and combine CG coefficients.

    Explanation
    ===========

    This function uses various symmetry and properties of sums and
    products of Clebsch-Gordan coefficients to simplify statements
    involving these terms [1]_.

    Examples
    ========

    Simplify the sum over CG(a,alpha,0,0,a,alpha) for all alpha to
    2*a+1

        >>> from sympy.physics.quantum.cg import CG, cg_simp
        >>> a = CG(1,1,0,0,1,1)
        >>> b = CG(1,0,0,0,1,0)
        >>> c = CG(1,-1,0,0,1,-1)
        >>> cg_simp(a+b+c)
        3

    See Also
    ========

    CG: Clebsh-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """
    if isinstance(e, Add):
        return _cg_simp_add(e)
    elif isinstance(e, Sum):
        return _cg_simp_sum(e)
    elif isinstance(e, Mul):
        return Mul(*[cg_simp(arg) for arg in e.args])
    elif isinstance(e, Pow):
        return Pow(cg_simp(e.base), e.exp)
    else:
        return e