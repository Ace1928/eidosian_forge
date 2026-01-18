from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def sign2map(a, var):
    """
    varname,ctype,atype
    init,init.r,init.i,pytype
    vardebuginfo,vardebugshowvalue,varshowvalue
    varrformat

    intent
    """
    out_a = a
    if isintent_out(var):
        for k in var['intent']:
            if k[:4] == 'out=':
                out_a = k[4:]
                break
    ret = {'varname': a, 'outvarname': out_a, 'ctype': getctype(var)}
    intent_flags = []
    for f, s in isintent_dict.items():
        if f(var):
            intent_flags.append('F2PY_%s' % s)
    if intent_flags:
        ret['intent'] = '|'.join(intent_flags)
    else:
        ret['intent'] = 'F2PY_INTENT_IN'
    if isarray(var):
        ret['varrformat'] = 'N'
    elif ret['ctype'] in c2buildvalue_map:
        ret['varrformat'] = c2buildvalue_map[ret['ctype']]
    else:
        ret['varrformat'] = 'O'
    ret['init'], ret['showinit'] = getinit(a, var)
    if hasinitvalue(var) and iscomplex(var) and (not isarray(var)):
        ret['init.r'], ret['init.i'] = markoutercomma(ret['init'][1:-1]).split('@,@')
    if isexternal(var):
        ret['cbnamekey'] = a
        if a in lcb_map:
            ret['cbname'] = lcb_map[a]
            ret['maxnofargs'] = lcb2_map[lcb_map[a]]['maxnofargs']
            ret['nofoptargs'] = lcb2_map[lcb_map[a]]['nofoptargs']
            ret['cbdocstr'] = lcb2_map[lcb_map[a]]['docstr']
            ret['cblatexdocstr'] = lcb2_map[lcb_map[a]]['latexdocstr']
        else:
            ret['cbname'] = a
            errmess('sign2map: Confused: external %s is not in lcb_map%s.\n' % (a, list(lcb_map.keys())))
    if isstring(var):
        ret['length'] = getstrlength(var)
    if isarray(var):
        ret = dictappend(ret, getarrdims(a, var))
        dim = copy.copy(var['dimension'])
    if ret['ctype'] in c2capi_map:
        ret['atype'] = c2capi_map[ret['ctype']]
        ret['elsize'] = get_elsize(var)
    if debugcapi(var):
        il = [isintent_in, 'input', isintent_out, 'output', isintent_inout, 'inoutput', isrequired, 'required', isoptional, 'optional', isintent_hide, 'hidden', iscomplex, 'complex scalar', l_and(isscalar, l_not(iscomplex)), 'scalar', isstring, 'string', isarray, 'array', iscomplexarray, 'complex array', isstringarray, 'string array', iscomplexfunction, 'complex function', l_and(isfunction, l_not(iscomplexfunction)), 'function', isexternal, 'callback', isintent_callback, 'callback', isintent_aux, 'auxiliary']
        rl = []
        for i in range(0, len(il), 2):
            if il[i](var):
                rl.append(il[i + 1])
        if isstring(var):
            rl.append('slen(%s)=%s' % (a, ret['length']))
        if isarray(var):
            ddim = ','.join(map(lambda x, y: '%s|%s' % (x, y), var['dimension'], dim))
            rl.append('dims(%s)' % ddim)
        if isexternal(var):
            ret['vardebuginfo'] = 'debug-capi:%s=>%s:%s' % (a, ret['cbname'], ','.join(rl))
        else:
            ret['vardebuginfo'] = 'debug-capi:%s %s=%s:%s' % (ret['ctype'], a, ret['showinit'], ','.join(rl))
        if isscalar(var):
            if ret['ctype'] in cformat_map:
                ret['vardebugshowvalue'] = 'debug-capi:%s=%s' % (a, cformat_map[ret['ctype']])
        if isstring(var):
            ret['vardebugshowvalue'] = 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"' % (a, a)
        if isexternal(var):
            ret['vardebugshowvalue'] = 'debug-capi:%s=%%p' % a
    if ret['ctype'] in cformat_map:
        ret['varshowvalue'] = '#name#:%s=%s' % (a, cformat_map[ret['ctype']])
        ret['showvalueformat'] = '%s' % cformat_map[ret['ctype']]
    if isstring(var):
        ret['varshowvalue'] = '#name#:slen(%s)=%%d %s=\\"%%s\\"' % (a, a)
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    if hasnote(var):
        ret['note'] = var['note']
    return ret