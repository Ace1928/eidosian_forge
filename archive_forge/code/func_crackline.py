import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def crackline(line, reset=0):
    """
    reset=-1  --- initialize
    reset=0   --- crack the line
    reset=1   --- final check if mismatch of blocks occurred

    Cracked data is saved in grouplist[0].
    """
    global beginpattern, groupcounter, groupname, groupcache, grouplist
    global filepositiontext, currentfilename, neededmodule, expectbegin
    global skipblocksuntil, skipemptyends, previous_context, gotnextfile
    _, has_semicolon = split_by_unquoted(line, ';')
    if has_semicolon and (not (f2pyenhancementspattern[0].match(line) or multilinepattern[0].match(line))):
        assert reset == 0, repr(reset)
        line, semicolon_line = split_by_unquoted(line, ';')
        while semicolon_line:
            crackline(line, reset)
            line, semicolon_line = split_by_unquoted(semicolon_line[1:], ';')
        crackline(line, reset)
        return
    if reset < 0:
        groupcounter = 0
        groupname = {groupcounter: ''}
        groupcache = {groupcounter: {}}
        grouplist = {groupcounter: []}
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['block'] = ''
        groupcache[groupcounter]['name'] = ''
        neededmodule = -1
        skipblocksuntil = -1
        return
    if reset > 0:
        fl = 0
        if f77modulename and neededmodule == groupcounter:
            fl = 2
        while groupcounter > fl:
            outmess('crackline: groupcounter=%s groupname=%s\n' % (repr(groupcounter), repr(groupname)))
            outmess('crackline: Mismatch of blocks encountered. Trying to fix it by assuming "end" statement.\n')
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1
        if f77modulename and neededmodule == groupcounter:
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1
            neededmodule = -1
        return
    if line == '':
        return
    flag = 0
    for pat in [dimensionpattern, externalpattern, intentpattern, optionalpattern, requiredpattern, parameterpattern, datapattern, publicpattern, privatepattern, intrinsicpattern, endifpattern, endpattern, formatpattern, beginpattern, functionpattern, subroutinepattern, implicitpattern, typespattern, commonpattern, callpattern, usepattern, containspattern, entrypattern, f2pyenhancementspattern, multilinepattern, moduleprocedurepattern]:
        m = pat[0].match(line)
        if m:
            break
        flag = flag + 1
    if not m:
        re_1 = crackline_re_1
        if 0 <= skipblocksuntil <= groupcounter:
            return
        if 'externals' in groupcache[groupcounter]:
            for name in groupcache[groupcounter]['externals']:
                if name in invbadnames:
                    name = invbadnames[name]
                if 'interfaced' in groupcache[groupcounter] and name in groupcache[groupcounter]['interfaced']:
                    continue
                m1 = re.match('(?P<before>[^"]*)\\b%s\\b\\s*@\\(@(?P<args>[^@]*)@\\)@.*\\Z' % name, markouterparen(line), re.I)
                if m1:
                    m2 = re_1.match(m1.group('before'))
                    a = _simplifyargs(m1.group('args'))
                    if m2:
                        line = 'callfun %s(%s) result (%s)' % (name, a, m2.group('result'))
                    else:
                        line = 'callfun %s(%s)' % (name, a)
                    m = callfunpattern[0].match(line)
                    if not m:
                        outmess('crackline: could not resolve function call for line=%s.\n' % repr(line))
                        return
                    analyzeline(m, 'callfun', line)
                    return
        if verbose > 1 or (verbose == 1 and currentfilename.lower().endswith('.pyf')):
            previous_context = None
            outmess('crackline:%d: No pattern for line\n' % groupcounter)
        return
    elif pat[1] == 'end':
        if 0 <= skipblocksuntil < groupcounter:
            groupcounter = groupcounter - 1
            if skipblocksuntil <= groupcounter:
                return
        if groupcounter <= 0:
            raise Exception('crackline: groupcounter(=%s) is nonpositive. Check the blocks.' % groupcounter)
        m1 = beginpattern[0].match(line)
        if m1 and (not m1.group('this') == groupname[groupcounter]):
            raise Exception('crackline: End group %s does not match with previous Begin group %s\n\t%s' % (repr(m1.group('this')), repr(groupname[groupcounter]), filepositiontext))
        if skipblocksuntil == groupcounter:
            skipblocksuntil = -1
        grouplist[groupcounter - 1].append(groupcache[groupcounter])
        grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
        del grouplist[groupcounter]
        groupcounter = groupcounter - 1
        if not skipemptyends:
            expectbegin = 1
    elif pat[1] == 'begin':
        if 0 <= skipblocksuntil <= groupcounter:
            groupcounter = groupcounter + 1
            return
        gotnextfile = 0
        analyzeline(m, pat[1], line)
        expectbegin = 0
    elif pat[1] == 'endif':
        pass
    elif pat[1] == 'moduleprocedure':
        analyzeline(m, pat[1], line)
    elif pat[1] == 'contains':
        if ignorecontains:
            return
        if 0 <= skipblocksuntil <= groupcounter:
            return
        skipblocksuntil = groupcounter
    else:
        if 0 <= skipblocksuntil <= groupcounter:
            return
        analyzeline(m, pat[1], line)