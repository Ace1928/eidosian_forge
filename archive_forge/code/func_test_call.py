import pytest
import rpy2.robjects as robjects
import array
def test_call():
    ri_f = rinterface.baseenv.find('sum')
    ro_f = robjects.Function(ri_f)
    ro_v = robjects.IntVector(array.array('i', [1, 2, 3]))
    s = ro_f(ro_v)
    assert s[0] == 6