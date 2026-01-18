import pytest
import rpy2.robjects as robjects
import array
def test_init_empty():
    env = robjects.Environment()
    assert env.typeof == rinterface.RTYPES.ENVSXP