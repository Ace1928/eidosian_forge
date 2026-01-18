from .auxfuncs import (
def buildusevar(name, realname, vars, usemodulename):
    outmess('\t\t\tConstructing wrapper function for variable "%s=>%s"...\n' % (name, realname))
    ret = {}
    vrd = {'name': name, 'realname': realname, 'REALNAME': realname.upper(), 'usemodulename': usemodulename, 'USEMODULENAME': usemodulename.upper(), 'texname': name.replace('_', '\\_'), 'begintitle': gentitle('%s=>%s' % (name, realname)), 'endtitle': gentitle('end of %s=>%s' % (name, realname)), 'apiname': '#modulename#_use_%s_from_%s' % (realname, usemodulename)}
    nummap = {0: 'Ro', 1: 'Ri', 2: 'Rii', 3: 'Riii', 4: 'Riv', 5: 'Rv', 6: 'Rvi', 7: 'Rvii', 8: 'Rviii', 9: 'Rix'}
    vrd['texnamename'] = name
    for i in nummap.keys():
        vrd['texnamename'] = vrd['texnamename'].replace(repr(i), nummap[i])
    if hasnote(vars[realname]):
        vrd['note'] = vars[realname]['note']
    rd = dictappend({}, vrd)
    print(name, realname, vars[realname])
    ret = applyrules(usemodule_rules, rd)
    return ret