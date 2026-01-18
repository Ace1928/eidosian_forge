from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices import DM
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMRankError, DMValueError, DMShapeError, DMDomainError
from sympy.polys.matrices.lll import _ddm_lll, ddm_lll, ddm_lll_transform
from sympy.testing.pytest import raises
def test_lll_wrong_domain():
    wrong_domain_matrix = DomainMatrix.ones((3, 3), QQ)
    raises(DMDomainError, lambda: _ddm_lll(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: ddm_lll(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.lll())
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.to_sdm().lll())
    raises(DMDomainError, lambda: wrong_domain_matrix.lll())
    raises(DMDomainError, lambda: _ddm_lll(wrong_domain_matrix.rep, return_transform=True))
    raises(DMDomainError, lambda: ddm_lll_transform(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.lll_transform())
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.to_sdm().lll_transform())
    raises(DMDomainError, lambda: wrong_domain_matrix.lll_transform())