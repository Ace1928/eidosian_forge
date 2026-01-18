from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def getpydocsign(a, var):
    global lcb_map
    if isfunction(var):
        if 'result' in var:
            af = var['result']
        else:
            af = var['name']
        if af in var['vars']:
            return getpydocsign(af, var['vars'][af])
        else:
            errmess('getctype: function %s has no return value?!\n' % af)
        return ('', '')
    sig, sigout = (a, a)
    opt = ''
    if isintent_in(var):
        opt = 'input'
    elif isintent_inout(var):
        opt = 'in/output'
    out_a = a
    if isintent_out(var):
        for k in var['intent']:
            if k[:4] == 'out=':
                out_a = k[4:]
                break
    init = ''
    ctype = getctype(var)
    if hasinitvalue(var):
        init, showinit = getinit(a, var)
        init = ', optional\\n    Default: %s' % showinit
    if isscalar(var):
        if isintent_inout(var):
            sig = "%s : %s rank-0 array(%s,'%s')%s" % (a, opt, c2py_map[ctype], c2pycode_map[ctype], init)
        else:
            sig = '%s : %s %s%s' % (a, opt, c2py_map[ctype], init)
        sigout = '%s : %s' % (out_a, c2py_map[ctype])
    elif isstring(var):
        if isintent_inout(var):
            sig = "%s : %s rank-0 array(string(len=%s),'c')%s" % (a, opt, getstrlength(var), init)
        else:
            sig = '%s : %s string(len=%s)%s' % (a, opt, getstrlength(var), init)
        sigout = '%s : string(len=%s)' % (out_a, getstrlength(var))
    elif isarray(var):
        dim = var['dimension']
        rank = repr(len(dim))
        sig = "%s : %s rank-%s array('%s') with bounds (%s)%s" % (a, opt, rank, c2pycode_map[ctype], ','.join(dim), init)
        if a == out_a:
            sigout = "%s : rank-%s array('%s') with bounds (%s)" % (a, rank, c2pycode_map[ctype], ','.join(dim))
        else:
            sigout = "%s : rank-%s array('%s') with bounds (%s) and %s storage" % (out_a, rank, c2pycode_map[ctype], ','.join(dim), a)
    elif isexternal(var):
        ua = ''
        if a in lcb_map and lcb_map[a] in lcb2_map and ('argname' in lcb2_map[lcb_map[a]]):
            ua = lcb2_map[lcb_map[a]]['argname']
            if not ua == a:
                ua = ' => %s' % ua
            else:
                ua = ''
        sig = '%s : call-back function%s' % (a, ua)
        sigout = sig
    else:
        errmess('getpydocsign: Could not resolve docsignature for "%s".\n' % a)
    return (sig, sigout)