from math import prod
from sympy.core.mul import Mul
from sympy.matrices.dense import (Matrix, diag)
from sympy.polys.polytools import (Poly, degree_list, rem)
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import IndexedBase
from sympy.polys.monomials import itermonomials, monomial_deg
from sympy.polys.orderings import monomial_key
from sympy.polys.polytools import poly_from_expr, total_degree
from sympy.functions.combinatorial.factorials import binomial
from itertools import combinations_with_replacement
from sympy.utilities.exceptions import sympy_deprecation_warning
def get_max_degrees(self, polynomial):
    """
        Returns a list of the maximum degree of each variable appearing
        in the coefficients of the Dixon polynomial. The coefficients are
        viewed as polys in $x_1, x_2, \\dots, x_n$.
        """
    deg_lists = [degree_list(Poly(poly, self.variables)) for poly in polynomial.coeffs()]
    max_degrees = [max(degs) for degs in zip(*deg_lists)]
    return max_degrees