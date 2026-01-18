import pytest
from rpy2.rinterface import NA_Character
from rpy2 import robjects
def test_factor_with_attrs():
    r_src = '\n    x <- factor(c("a","b","a"))\n    attr(x, "foo") <- "bar"\n    x\n    '
    x = robjects.r(r_src)
    assert 'foo' in x.list_attrs()