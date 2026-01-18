import sys
import copy
from . import __version__
def buildcfuncs():
    from .capi_maps import c2capi_map
    for k in c2capi_map.keys():
        m = 'pyarr_from_p_%s1' % k
        cppmacros[m] = '#define %s(v) (PyArray_SimpleNewFromData(0,NULL,%s,(char *)v))' % (m, c2capi_map[k])
    k = 'string'
    m = 'pyarr_from_p_%s1' % k
    cppmacros[m] = '#define %s(v,dims) (PyArray_New(&PyArray_Type, 1, dims, NPY_STRING, NULL, v, 1, NPY_ARRAY_CARRAY, NULL))' % m