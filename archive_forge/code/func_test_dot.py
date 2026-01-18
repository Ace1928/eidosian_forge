import pytest
import rpy2.robjects as robjects
import array
def test_dot():
    m = robjects.r.matrix(robjects.IntVector(range(4)), nrow=2, ncol=2)
    m2 = m.dot(m)
    assert tuple(m2) == (2, 3, 6, 11)