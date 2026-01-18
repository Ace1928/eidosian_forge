from ctypes import POINTER, c_bool, c_char_p, c_double, c_int, c_int64, c_void_p
from functools import partial
from django.contrib.gis.gdal.prototypes.errcheck import (
def const_string_output(func, argtypes, offset=None, decoding=None, cpl=False):
    func.argtypes = argtypes
    if offset:
        func.restype = c_int
    else:
        func.restype = c_char_p

    def _check_const(result, func, cargs):
        res = check_const_string(result, func, cargs, offset=offset, cpl=cpl)
        if res and decoding:
            res = res.decode(decoding)
        return res
    func.errcheck = _check_const
    return func