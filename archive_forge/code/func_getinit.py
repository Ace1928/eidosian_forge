from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def getinit(a, var):
    if isstring(var):
        init, showinit = ('""', "''")
    else:
        init, showinit = ('', '')
    if hasinitvalue(var):
        init = var['=']
        showinit = init
        if iscomplex(var) or iscomplexarray(var):
            ret = {}
            try:
                v = var['=']
                if ',' in v:
                    ret['init.r'], ret['init.i'] = markoutercomma(v[1:-1]).split('@,@')
                else:
                    v = eval(v, {}, {})
                    ret['init.r'], ret['init.i'] = (str(v.real), str(v.imag))
            except Exception:
                raise ValueError("getinit: expected complex number `(r,i)' but got `%s' as initial value of %r." % (init, a))
            if isarray(var):
                init = '(capi_c.r=%s,capi_c.i=%s,capi_c)' % (ret['init.r'], ret['init.i'])
        elif isstring(var):
            if not init:
                init, showinit = ('""', "''")
            if init[0] == "'":
                init = '"%s"' % init[1:-1].replace('"', '\\"')
            if init[0] == '"':
                showinit = "'%s'" % init[1:-1]
    return (init, showinit)