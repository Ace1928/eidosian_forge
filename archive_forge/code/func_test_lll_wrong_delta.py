from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices import DM
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMRankError, DMValueError, DMShapeError, DMDomainError
from sympy.polys.matrices.lll import _ddm_lll, ddm_lll, ddm_lll_transform
from sympy.testing.pytest import raises
def test_lll_wrong_delta():
    dummy_matrix = DomainMatrix.ones((3, 3), ZZ)
    for wrong_delta in [QQ(-1, 4), QQ(0, 1), QQ(1, 4), QQ(1, 1), QQ(100, 1)]:
        raises(DMValueError, lambda: _ddm_lll(dummy_matrix.rep, delta=wrong_delta))
        raises(DMValueError, lambda: ddm_lll(dummy_matrix.rep, delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.lll(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.to_sdm().lll(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.lll(delta=wrong_delta))
        raises(DMValueError, lambda: _ddm_lll(dummy_matrix.rep, delta=wrong_delta, return_transform=True))
        raises(DMValueError, lambda: ddm_lll_transform(dummy_matrix.rep, delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.lll_transform(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.to_sdm().lll_transform(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.lll_transform(delta=wrong_delta))