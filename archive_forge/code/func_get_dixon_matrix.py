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
def get_dixon_matrix(self, polynomial):
    """
        Construct the Dixon matrix from the coefficients of polynomial
        \\alpha. Each coefficient is viewed as a polynomial of x_1, ...,
        x_n.
        """
    max_degrees = self.get_max_degrees(polynomial)
    monomials = itermonomials(self.variables, max_degrees)
    monomials = sorted(monomials, reverse=True, key=monomial_key('lex', self.variables))
    dixon_matrix = Matrix([[Poly(c, *self.variables).coeff_monomial(m) for m in monomials] for c in polynomial.coeffs()])
    if dixon_matrix.shape[0] != dixon_matrix.shape[1]:
        keep = [column for column in range(dixon_matrix.shape[-1]) if any((element != 0 for element in dixon_matrix[:, column]))]
        dixon_matrix = dixon_matrix[:, keep]
    return dixon_matrix