import pytest
import rpy2.robjects as robjects
import array
def test_matmul():
    m = robjects.r.matrix(robjects.IntVector(range(1, 5)), nrow=2)
    m2 = m @ m
    for i, val in enumerate((7.0, 10.0, 15.0, 22.0)):
        assert m2[i] == val