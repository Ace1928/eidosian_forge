from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doClef(field):
    if re.search('perc|map', field):
        r = re.search('(perc|map)\\s*=\\s*(\\S*)', field)
        s.percVoice = 0 if r and r.group(2) not in ['on', 'true', 'perc'] else 1
        field = re.sub('(perc|map)\\s*=\\s*(\\S*)', '', field)
    clef, gtrans = (0, 0)
    clefn = re.search('alto1|alto2|alto4|alto|tenor|bass3|bass|treble|perc|none|tab', field)
    clefm = re.search("(?:^m=| m=|middle=)([A-Ga-g])([,']*)", field)
    trans_oct2 = re.search('octave=([-+]?\\d)', field)
    trans = re.search('(?:^t=| t=|transpose=)(-?[\\d]+)', field)
    trans_oct = re.search('([+-^_])(8|15)', field)
    cue_onoff = re.search('cue=(on|off)', field)
    strings = re.search('strings=(\\S+)', field)
    stafflines = re.search('stafflines=\\s*(\\d)', field)
    capo = re.search('capo=(\\d+)', field)
    if clefn:
        clef = clefn.group()
    if clefm:
        note, octstr = clefm.groups()
        nUp = note.upper()
        octnum = (4 if nUp == note else 5) + (len(octstr) if "'" in octstr else -len(octstr))
        gtrans = (3 if nUp in 'AFD' else 4) - octnum
        if clef not in ['perc', 'none']:
            clef = s.clefLineMap[nUp]
    if clef:
        s.gtrans = gtrans
        if clef != 'none':
            s.curClef = clef
        sign, line = s.clefMap[clef]
        if not sign:
            return
        c = E.Element('clef')
        if gstaff:
            c.set('number', str(gstaff))
        addElemT(c, 'sign', sign, lev + 2)
        if line:
            addElemT(c, 'line', line, lev + 2)
        if trans_oct:
            n = trans_oct.group(1) in '-_' and -1 or 1
            if trans_oct.group(2) == '15':
                n *= 2
            addElemT(c, 'clef-octave-change', str(n), lev + 2)
            if trans_oct.group(1) in '+-':
                s.gtrans += n
        atts.append((7, c))
    if trans_oct2:
        n = int(trans_oct2.group(1))
        s.gtrans = gtrans + n
    if trans != None:
        e = E.Element('transpose')
        addElemT(e, 'chromatic', str(trans.group(1)), lev + 3)
        atts.append((9, e))
    if cue_onoff:
        s.gcue_on = cue_onoff.group(1) == 'on'
    nlines = 0
    if clef == 'tab':
        s.tabStaff = s.pid
        if capo:
            s.capo = int(capo.group(1))
        if strings:
            s.tuning = strings.group(1).split(',')
        s.tunmid = [int(boct) * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(bstep)] + 12 + s.capo for bstep, boct in s.tuning]
        s.tunTup = sorted(zip(s.tunmid, range(len(s.tunmid), 0, -1)), reverse=1)
        s.tunmid.reverse()
        nlines = str(len(s.tuning))
        s.strAlloc.setlines(len(s.tuning), s.pid)
        s.nostems = 'nostems' in field
        s.diafret = 'diafret' in field
    if stafflines or nlines:
        e = E.Element('staff-details')
        if gstaff:
            e.set('number', str(gstaff))
        if not nlines:
            nlines = stafflines.group(1)
        addElemT(e, 'staff-lines', nlines, lev + 2)
        if clef == 'tab':
            for line, t in enumerate(s.tuning):
                st = E.Element('staff-tuning', line=str(line + 1))
                addElemT(st, 'tuning-step', t[0], lev + 3)
                addElemT(st, 'tuning-octave', t[1], lev + 3)
                addElem(e, st, lev + 2)
        if s.capo:
            addElemT(e, 'capo', str(s.capo), lev + 2)
        atts.append((8, e))