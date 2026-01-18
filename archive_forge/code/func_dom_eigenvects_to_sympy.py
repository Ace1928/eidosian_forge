from sympy.core.symbol import Dummy
from ..agca.extensions import FiniteExtension
from ..factortools import dup_factor_list
from ..polyroots import roots
from ..polytools import Poly
from ..rootoftools import CRootOf
from .domainmatrix import DomainMatrix
def dom_eigenvects_to_sympy(rational_eigenvects, algebraic_eigenvects, Matrix, **kwargs):
    result = []
    for field, eigenvalue, multiplicity, eigenvects in rational_eigenvects:
        eigenvects = eigenvects.rep.to_ddm()
        eigenvalue = field.to_sympy(eigenvalue)
        new_eigenvects = [Matrix([field.to_sympy(x) for x in vect]) for vect in eigenvects]
        result.append((eigenvalue, multiplicity, new_eigenvects))
    for field, minpoly, multiplicity, eigenvects in algebraic_eigenvects:
        eigenvects = eigenvects.rep.to_ddm()
        l = minpoly.gens[0]
        eigenvects = [[field.to_sympy(x) for x in vect] for vect in eigenvects]
        degree = minpoly.degree()
        minpoly = minpoly.as_expr()
        eigenvals = roots(minpoly, l, **kwargs)
        if len(eigenvals) != degree:
            eigenvals = [CRootOf(minpoly, l, idx) for idx in range(degree)]
        for eigenvalue in eigenvals:
            new_eigenvects = [Matrix([x.subs(l, eigenvalue) for x in vect]) for vect in eigenvects]
            result.append((eigenvalue, multiplicity, new_eigenvects))
    return result