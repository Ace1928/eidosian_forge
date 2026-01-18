from patsy.contrasts import ContrastMatrix, Treatment, Poly, Sum, Helmert, Diff
from patsy.categorical import C
from patsy.state import center, standardize, scale
from patsy.splines import bs
from patsy.mgcv_cubic_splines import cr, cc, te
def test_Q():
    a = 1
    assert Q('a') == 1
    assert Q('Q') is Q
    import pytest
    pytest.raises(NameError, Q, 'asdfsadfdsad')