import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_atomic_vector_to_numpy(self):
    v = robjects.vectors.IntVector((1, 2, 3))
    with rpyn.converter.context() as cv:
        a = cv.rpy2py(v)
    assert isinstance(a, numpy.ndarray)
    assert v[0] == 1