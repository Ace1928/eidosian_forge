from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
def test_dom_eigenvects_rational():
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(1), QQ(2)]], (2, 2), QQ)
    rational_eigenvects = [(QQ, QQ(3), 1, DomainMatrix([[QQ(1), QQ(1)]], (1, 2), QQ)), (QQ, QQ(0), 1, DomainMatrix([[QQ(-2), QQ(1)]], (1, 2), QQ))]
    assert dom_eigenvects(A) == (rational_eigenvects, [])
    sympy_eigenvects = [(S(3), 1, [Matrix([1, 1])]), (S(0), 1, [Matrix([-2, 1])])]
    assert dom_eigenvects_to_sympy(rational_eigenvects, [], Matrix) == sympy_eigenvects