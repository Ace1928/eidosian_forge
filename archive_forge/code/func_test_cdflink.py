import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import pytest
import statsmodels.genmod.families as families
from statsmodels.tools import numdiff as nd
@pytest.mark.parametrize('m', CasesCDFLink.methods)
@pytest.mark.parametrize('link1, link2', CasesCDFLink.link_pairs)
def test_cdflink(m, link1, link2):
    p = CasesCDFLink.p
    res1 = getattr(link1, m)(p)
    res2 = getattr(link2, m)(p)
    assert_allclose(res1, res2, atol=1e-08, rtol=1e-08)