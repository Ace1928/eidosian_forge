from __future__ import annotations
from sympy.vector.basisdependent import (BasisDependent, BasisDependentAdd,
from sympy.core import S, Pow
from sympy.core.expr import AtomicExpr
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
import sympy.vector
@property
def base_dyadic(self):
    """ The BaseDyadic involved in the product. """
    return self._base_instance