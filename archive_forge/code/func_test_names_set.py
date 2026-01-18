import pytest
import rpy2.robjects as robjects
import array
def test_names_set():
    dimnames = robjects.r.list(robjects.StrVector(['a', 'b', 'c']), robjects.StrVector(['d', 'e']))
    m = robjects.r.matrix(1, nrow=3, ncol=2)
    a = robjects.vectors.FloatArray(m)
    a.names = dimnames
    res = a.names
    r_identical = robjects.r.identical
    assert r_identical(dimnames[0], res[0])[0]
    assert r_identical(dimnames[1], res[1])[0]