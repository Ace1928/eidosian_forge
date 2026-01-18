from sympy.core.function import expand_log
from sympy.core.singleton import S
from sympy.core.symbol import Wild
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min)
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.assumptions import Q, ask
from sympy.codegen.cfunctions import log1p, log2, exp2, expm1
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core.expr import UnevaluatedExpr
from sympy.core.power import Pow
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.core.mul import Mul
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.utilities.iterables import sift
def replace_in_Add(self, e):
    """ passed as second argument to Basic.replace(...) """
    numsum, terms_with_func, other_non_num_terms = self._group_Add_terms(e)
    if numsum == 0:
        return e
    substituted, untouched = ([], [])
    for with_func in terms_with_func:
        if with_func.is_Mul:
            func, coeff = sift(with_func.args, lambda arg: arg.func == self.func, binary=True)
            if len(func) == 1 and len(coeff) == 1:
                func, coeff = (func[0], coeff[0])
            else:
                coeff = None
        elif with_func.func == self.func:
            func, coeff = (with_func, S.One)
        else:
            coeff = None
        if coeff is not None and coeff.is_number and (sign(coeff) == -sign(numsum)):
            if self.opportunistic:
                do_substitute = abs(coeff + numsum) < abs(numsum)
            else:
                do_substitute = coeff + numsum == 0
            if do_substitute:
                numsum += coeff
                substituted.append(coeff * self.func_m_1(*func.args))
                continue
        untouched.append(with_func)
    return e.func(numsum, *substituted, *untouched, *other_non_num_terms)