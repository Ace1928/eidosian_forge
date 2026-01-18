import pytest
import rpy2.robjects as robjects
import array
def test_eigen():
    m = robjects.r.matrix(robjects.IntVector((1, -1, -1, 1)), nrow=2)
    res = m.eigen()
    for i, val in enumerate(res.rx2('values')):
        assert (2, 0)[i] == val