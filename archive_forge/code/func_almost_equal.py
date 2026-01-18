import pytest
import rpy2.robjects as robjects
import array
def almost_equal(x, y, epsilon=1e-05):
    return abs(y - x) <= epsilon