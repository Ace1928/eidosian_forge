from ctypes import POINTER, c_bool, c_char_p, c_double, c_int, c_int64, c_void_p
from functools import partial
from django.contrib.gis.gdal.prototypes.errcheck import (
def srs_output(func, argtypes):
    """
    Generate a ctypes prototype for the given function with
    the given C arguments that returns a pointer to an OGR
    Spatial Reference System.
    """
    func.argtypes = argtypes
    func.restype = c_void_p
    func.errcheck = check_srs
    return func