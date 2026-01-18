from ctypes import POINTER, c_bool, c_char_p, c_double, c_int, c_int64, c_void_p
from functools import partial
from django.contrib.gis.gdal.prototypes.errcheck import (
def geom_output(func, argtypes, offset=None):
    """
    Generate a function that returns a Geometry either by reference
    or directly (if the return_geom keyword is set to True).
    """
    func.argtypes = argtypes
    if not offset:
        func.restype = c_void_p
        func.errcheck = check_geom
    else:
        func.restype = c_int

        def geomerrcheck(result, func, cargs):
            return check_geom_offset(result, func, cargs, offset)
        func.errcheck = geomerrcheck
    return func