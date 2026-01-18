import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_bad_array(self):
    u = numpy.array([1, 2, 3], dtype=numpy.uint32)
    with pytest.raises(ValueError):
        with (robjects.default_converter + rpyn.converter).context() as cv:
            cv.py2rpy(u)