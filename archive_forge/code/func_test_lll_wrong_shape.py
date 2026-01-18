from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices import DM
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMRankError, DMValueError, DMShapeError, DMDomainError
from sympy.polys.matrices.lll import _ddm_lll, ddm_lll, ddm_lll_transform
from sympy.testing.pytest import raises
def test_lll_wrong_shape():
    wrong_shape_matrix = DomainMatrix.ones((4, 3), ZZ)
    raises(DMShapeError, lambda: _ddm_lll(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: ddm_lll(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.lll())
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.to_sdm().lll())
    raises(DMShapeError, lambda: wrong_shape_matrix.lll())
    raises(DMShapeError, lambda: _ddm_lll(wrong_shape_matrix.rep, return_transform=True))
    raises(DMShapeError, lambda: ddm_lll_transform(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.lll_transform())
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.to_sdm().lll_transform())
    raises(DMShapeError, lambda: wrong_shape_matrix.lll_transform())