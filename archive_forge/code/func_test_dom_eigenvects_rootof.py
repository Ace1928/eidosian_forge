from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
def test_dom_eigenvects_rootof():
    A = DomainMatrix([[0, 0, 0, 0, -1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], (5, 5), QQ)
    Avects = dom_eigenvects(A)
    lamda = Avects[1][0][1].gens[0]
    irreducible = Poly(lamda ** 5 - lamda + 1, lamda, domain=QQ)
    K = FiniteExtension(irreducible)
    KK = K.from_sympy
    algebraic_eigenvects = [(K, irreducible, 1, DomainMatrix([[KK(lamda ** 4 - 1), KK(lamda ** 3), KK(lamda ** 2), KK(lamda), KK(1)]], (1, 5), K))]
    assert Avects == ([], algebraic_eigenvects)
    l0, l1, l2, l3, l4 = [CRootOf(lamda ** 5 - lamda + 1, i) for i in range(5)]
    sympy_eigenvects = [(l0, 1, [Matrix([-1 + l0 ** 4, l0 ** 3, l0 ** 2, l0, 1])]), (l1, 1, [Matrix([-1 + l1 ** 4, l1 ** 3, l1 ** 2, l1, 1])]), (l2, 1, [Matrix([-1 + l2 ** 4, l2 ** 3, l2 ** 2, l2, 1])]), (l3, 1, [Matrix([-1 + l3 ** 4, l3 ** 3, l3 ** 2, l3, 1])]), (l4, 1, [Matrix([-1 + l4 ** 4, l4 ** 3, l4 ** 2, l4, 1])])]
    assert dom_eigenvects_to_sympy([], algebraic_eigenvects, Matrix) == sympy_eigenvects