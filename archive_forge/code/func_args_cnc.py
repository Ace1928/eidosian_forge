from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
from sympy.core import Basic, sympify, S
from sympy.core.mul import mul, Mul
from sympy.core.numbers import Number, Integer
from sympy.core.symbol import Dummy
from sympy.functions import adjoint
from sympy.strategies import (rm_id, unpack, typed, flatten, exhaust,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices.matrices import MatrixBase
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.matrices.expressions._shape import validate_matmul_integer as validate
from .inverse import Inverse
from .matexpr import MatrixExpr
from .matpow import MatPow
from .transpose import transpose
from .permutation import PermutationMatrix
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix
def args_cnc(self, cset=False, warn=True, **kwargs):
    coeff_c = [x for x in self.args if x.is_commutative]
    coeff_nc = [x for x in self.args if not x.is_commutative]
    if cset:
        clen = len(coeff_c)
        coeff_c = set(coeff_c)
        if clen and warn and (len(coeff_c) != clen):
            raise ValueError('repeated commutative arguments: %s' % [ci for ci in coeff_c if list(self.args).count(ci) > 1])
    return [coeff_c, coeff_nc]