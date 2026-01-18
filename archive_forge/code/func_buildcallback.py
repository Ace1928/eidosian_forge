from . import __version__
from .auxfuncs import (
from . import cfuncs
def buildcallback(rout, um):
    from . import capi_maps
    outmess('    Constructing call-back function "cb_%s_in_%s"\n' % (rout['name'], um))
    args, depargs = getargs(rout)
    capi_maps.depargs = depargs
    var = rout['vars']
    vrd = capi_maps.cb_routsign2map(rout, um)
    rd = dictappend({}, vrd)
    cb_map[um].append([rout['name'], rd['name']])
    for r in cb_rout_rules:
        if '_check' in r and r['_check'](rout) or '_check' not in r:
            ar = applyrules(r, vrd, rout)
            rd = dictappend(rd, ar)
    savevrd = {}
    for i, a in enumerate(args):
        vrd = capi_maps.cb_sign2map(a, var[a], index=i)
        savevrd[a] = vrd
        for r in cb_arg_rules:
            if '_depend' in r:
                continue
            if '_optional' in r and isoptional(var[a]):
                continue
            if '_check' in r and r['_check'](var[a]) or '_check' not in r:
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    for a in args:
        vrd = savevrd[a]
        for r in cb_arg_rules:
            if '_depend' in r:
                continue
            if '_optional' not in r or ('_optional' in r and isrequired(var[a])):
                continue
            if '_check' in r and r['_check'](var[a]) or '_check' not in r:
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    for a in depargs:
        vrd = savevrd[a]
        for r in cb_arg_rules:
            if '_depend' not in r:
                continue
            if '_optional' in r:
                continue
            if '_check' in r and r['_check'](var[a]) or '_check' not in r:
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    if 'args' in rd and 'optargs' in rd:
        if isinstance(rd['optargs'], list):
            rd['optargs'] = rd['optargs'] + ['\n#ifndef F2PY_CB_RETURNCOMPLEX\n,\n#endif\n']
            rd['optargs_nm'] = rd['optargs_nm'] + ['\n#ifndef F2PY_CB_RETURNCOMPLEX\n,\n#endif\n']
            rd['optargs_td'] = rd['optargs_td'] + ['\n#ifndef F2PY_CB_RETURNCOMPLEX\n,\n#endif\n']
    if isinstance(rd['docreturn'], list):
        rd['docreturn'] = stripcomma(replace('#docreturn#', {'docreturn': rd['docreturn']}))
    optargs = stripcomma(replace('#docsignopt#', {'docsignopt': rd['docsignopt']}))
    if optargs == '':
        rd['docsignature'] = stripcomma(replace('#docsign#', {'docsign': rd['docsign']}))
    else:
        rd['docsignature'] = replace('#docsign#[#docsignopt#]', {'docsign': rd['docsign'], 'docsignopt': optargs})
    rd['latexdocsignature'] = rd['docsignature'].replace('_', '\\_')
    rd['latexdocsignature'] = rd['latexdocsignature'].replace(',', ', ')
    rd['docstrsigns'] = []
    rd['latexdocstrsigns'] = []
    for k in ['docstrreq', 'docstropt', 'docstrout', 'docstrcbs']:
        if k in rd and isinstance(rd[k], list):
            rd['docstrsigns'] = rd['docstrsigns'] + rd[k]
        k = 'latex' + k
        if k in rd and isinstance(rd[k], list):
            rd['latexdocstrsigns'] = rd['latexdocstrsigns'] + rd[k][0:1] + ['\\begin{description}'] + rd[k][1:] + ['\\end{description}']
    if 'args' not in rd:
        rd['args'] = ''
        rd['args_td'] = ''
        rd['args_nm'] = ''
    if not (rd.get('args') or rd.get('optargs') or rd.get('strarglens')):
        rd['noargs'] = 'void'
    ar = applyrules(cb_routine_rules, rd)
    cfuncs.callbacks[rd['name']] = ar['body']
    if isinstance(ar['need'], str):
        ar['need'] = [ar['need']]
    if 'need' in rd:
        for t in cfuncs.typedefs.keys():
            if t in rd['need']:
                ar['need'].append(t)
    cfuncs.typedefs_generated[rd['name'] + '_typedef'] = ar['cbtypedefs']
    ar['need'].append(rd['name'] + '_typedef')
    cfuncs.needs[rd['name']] = ar['need']
    capi_maps.lcb2_map[rd['name']] = {'maxnofargs': ar['maxnofargs'], 'nofoptargs': ar['nofoptargs'], 'docstr': ar['docstr'], 'latexdocstr': ar['latexdocstr'], 'argname': rd['argname']}
    outmess('      %s\n' % ar['docstrshort'])
    return