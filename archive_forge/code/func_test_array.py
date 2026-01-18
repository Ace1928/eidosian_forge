import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_array(self):
    i2d = numpy.array([[1, 2, 3], [4, 5, 6]], dtype='i')
    with (robjects.default_converter + rpyn.converter).context() as cv:
        i2d_r = cv.py2rpy(i2d)
    assert r['storage.mode'](i2d_r)[0] == 'integer'
    assert tuple(r['dim'](i2d_r)) == (2, 3)
    assert r['['](i2d_r, 1, 2)[0] == i2d[0, 1]
    f3d = numpy.arange(24, dtype='f').reshape((2, 3, 4))
    with (robjects.default_converter + rpyn.converter).context() as cv:
        f3d_r = cv.py2rpy(f3d)
    assert r['storage.mode'](f3d_r)[0] == 'double'
    assert tuple(r['dim'](f3d_r)) == (2, 3, 4)