import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def check_homogeneous(self, obj, mode, storage_mode):
    with (robjects.default_converter + rpyn.converter).context() as cv:
        converted = cv.py2rpy(obj)
    assert r['mode'](converted)[0] == mode
    assert r['storage.mode'](converted)[0] == storage_mode
    assert list(obj) == list(converted)
    assert r['is.array'](converted)[0] is True
    return converted