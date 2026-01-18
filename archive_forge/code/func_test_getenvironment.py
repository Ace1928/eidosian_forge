import pytest
import rpy2.robjects as robjects
def test_getenvironment():
    fml = robjects.Formula('y ~ x')
    env = fml.getenvironment()
    assert env.rclass[0] == 'environment'