import array
import pytest
import rpy2.robjects as robjects
def test_eval():
    x = robjects.baseenv['seq'](1, 50, 2)
    res = robjects.r('sum(%s)' % x.r_repr())
    assert res[0] == 625