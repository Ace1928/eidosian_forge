from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import oo, equal_valued
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.printing.pretty.stringpict import stringPict
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
@property
def is_commutative(self):
    """
        Override Function's is_commutative so that order is preserved in
        represented expressions
        """
    return False