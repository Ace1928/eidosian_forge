from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def getstrlength(var):
    if isstringfunction(var):
        if 'result' in var:
            a = var['result']
        else:
            a = var['name']
        if a in var['vars']:
            return getstrlength(var['vars'][a])
        else:
            errmess('getstrlength: function %s has no return value?!\n' % a)
    if not isstring(var):
        errmess('getstrlength: expected a signature of a string but got: %s\n' % repr(var))
    len = '1'
    if 'charselector' in var:
        a = var['charselector']
        if '*' in a:
            len = a['*']
        elif 'len' in a:
            len = f2cexpr(a['len'])
    if re.match('\\(\\s*(\\*|:)\\s*\\)', len) or re.match('(\\*|:)', len):
        if isintent_hide(var):
            errmess('getstrlength:intent(hide): expected a string with defined length but got: %s\n' % repr(var))
        len = '-1'
    return len