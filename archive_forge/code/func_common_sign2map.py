from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def common_sign2map(a, var):
    ret = {'varname': a, 'ctype': getctype(var)}
    if isstringarray(var):
        ret['ctype'] = 'char'
    if ret['ctype'] in c2capi_map:
        ret['atype'] = c2capi_map[ret['ctype']]
        ret['elsize'] = get_elsize(var)
    if ret['ctype'] in cformat_map:
        ret['showvalueformat'] = '%s' % cformat_map[ret['ctype']]
    if isarray(var):
        ret = dictappend(ret, getarrdims(a, var))
    elif isstring(var):
        ret['size'] = getstrlength(var)
        ret['rank'] = '1'
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    if hasnote(var):
        ret['note'] = var['note']
        var['note'] = ['See elsewhere.']
    ret['arrdocstr'] = getarrdocsign(a, var)
    return ret