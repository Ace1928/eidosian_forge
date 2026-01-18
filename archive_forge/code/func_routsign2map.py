from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def routsign2map(rout):
    """
    name,NAME,begintitle,endtitle
    rname,ctype,rformat
    routdebugshowvalue
    """
    global lcb_map
    name = rout['name']
    fname = getfortranname(rout)
    ret = {'name': name, 'texname': name.replace('_', '\\_'), 'name_lower': name.lower(), 'NAME': name.upper(), 'begintitle': gentitle(name), 'endtitle': gentitle('end of %s' % name), 'fortranname': fname, 'FORTRANNAME': fname.upper(), 'callstatement': getcallstatement(rout) or '', 'usercode': getusercode(rout) or '', 'usercode1': getusercode1(rout) or ''}
    if '_' in fname:
        ret['F_FUNC'] = 'F_FUNC_US'
    else:
        ret['F_FUNC'] = 'F_FUNC'
    if '_' in name:
        ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC_US'
    else:
        ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC'
    lcb_map = {}
    if 'use' in rout:
        for u in rout['use'].keys():
            if u in cb_rules.cb_map:
                for un in cb_rules.cb_map[u]:
                    ln = un[0]
                    if 'map' in rout['use'][u]:
                        for k in rout['use'][u]['map'].keys():
                            if rout['use'][u]['map'][k] == un[0]:
                                ln = k
                                break
                    lcb_map[ln] = un[1]
    elif 'externals' in rout and rout['externals']:
        errmess('routsign2map: Confused: function %s has externals %s but no "use" statement.\n' % (ret['name'], repr(rout['externals'])))
    ret['callprotoargument'] = getcallprotoargument(rout, lcb_map) or ''
    if isfunction(rout):
        if 'result' in rout:
            a = rout['result']
        else:
            a = rout['name']
        ret['rname'] = a
        ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, rout)
        ret['ctype'] = getctype(rout['vars'][a])
        if hasresultnote(rout):
            ret['resultnote'] = rout['vars'][a]['note']
            rout['vars'][a]['note'] = ['See elsewhere.']
        if ret['ctype'] in c2buildvalue_map:
            ret['rformat'] = c2buildvalue_map[ret['ctype']]
        else:
            ret['rformat'] = 'O'
            errmess('routsign2map: no c2buildvalue key for type %s\n' % repr(ret['ctype']))
        if debugcapi(rout):
            if ret['ctype'] in cformat_map:
                ret['routdebugshowvalue'] = 'debug-capi:%s=%s' % (a, cformat_map[ret['ctype']])
            if isstringfunction(rout):
                ret['routdebugshowvalue'] = 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"' % (a, a)
        if isstringfunction(rout):
            ret['rlength'] = getstrlength(rout['vars'][a])
            if ret['rlength'] == '-1':
                errmess('routsign2map: expected explicit specification of the length of the string returned by the fortran function %s; taking 10.\n' % repr(rout['name']))
                ret['rlength'] = '10'
    if hasnote(rout):
        ret['note'] = rout['note']
        rout['note'] = ['See elsewhere.']
    return ret