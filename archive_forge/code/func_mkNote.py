from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkNote(s, n, lev):
    isgrace = getattr(n, 'grace', '')
    ischord = getattr(n, 'chord', '')
    if s.ntup >= 0 and (not isgrace) and (not ischord):
        s.ntup -= 1
        if s.ntup == -1 and s.trem <= 0:
            s.intrem = 0
    nnum, nden = n.dur.t
    if s.intrem:
        nnum += nnum
    if nden == 0:
        nden = 1
    num, den = simplify(nnum * s.unitLcur[0], nden * s.unitLcur[1])
    if den > 64:
        num = int(round(64 * float(num) / den))
        num, den = simplify(max([num, 1]), 64)
        info('duration too small: rounded to %d/%d' % (num, den))
    if n.name == 'rest' and ('Z' in n.t or 'X' in n.t):
        num, den = s.mdur
    noMsrRest = not (n.name == 'rest' and (num, den) == s.mdur)
    dvs = 4 * s.divisions * num // den
    rdvs = dvs
    num, den = simplify(num, den * 4)
    ndot = 0
    if num == 3 and noMsrRest:
        ndot = 1
        den = den // 2
    if num == 7 and noMsrRest:
        ndot = 2
        den = den // 4
    nt = E.Element('note')
    if isgrace:
        grace = E.Element('grace')
        if s.acciatura:
            grace.set('slash', 'yes')
            s.acciatura = 0
        addElem(nt, grace, lev + 1)
        dvs = rdvs = 0
        if den <= 16:
            den = 32
    if s.gcue_on:
        cue = E.Element('cue')
        addElem(nt, cue, lev + 1)
    if ischord:
        chord = E.Element('chord')
        addElem(nt, chord, lev + 1)
        rdvs = 0
    if den not in s.typeMap:
        info('illegal duration %d/%d' % (nnum, nden))
        den = min((x for x in s.typeMap.keys() if x > den))
    xmltype = str(s.typeMap[den])
    acc, step, oct = ('', 'C', '0')
    alter, midi, notehead = ('', '', '')
    if n.name == 'rest':
        if 'x' in n.t or 'X' in n.t:
            nt.set('print-object', 'no')
        rest = E.Element('rest')
        if not noMsrRest:
            rest.set('measure', 'yes')
        addElem(nt, rest, lev + 1)
    else:
        p = n.pitch.t
        if len(p) == 3:
            acc, step, oct = p
        else:
            step, oct = p
        pitch, alter, midi, notehead = s.mkPitch(acc, step, oct, lev + 1)
        if midi:
            acc = ''
        addElem(nt, pitch, lev + 1)
    if s.ntup >= 0:
        dvs = dvs * s.tmden // s.tmnum
    if dvs:
        addElemT(nt, 'duration', str(dvs), lev + 1)
        if not ischord:
            s.gTime = (s.gTime[1], s.gTime[1] + dvs)
    ptup = (step, oct)
    tstop = ptup in s.ties and s.ties[ptup][2] == s.overlayVnum
    if tstop:
        tie = E.Element('tie', type='stop')
        addElem(nt, tie, lev + 1)
    if getattr(n, 'tie', 0):
        tie = E.Element('tie', type='start')
        addElem(nt, tie, lev + 1)
    if (s.midprg != ['', '', '', ''] or midi) and n.name != 'rest':
        instId = 'I%s-%s' % (s.pid, 'X' + midi if midi else s.vid)
        chan, midi = ('10', midi) if midi else s.midprg[:2]
        inst = E.Element('instrument', id=instId)
        addElem(nt, inst, lev + 1)
        if instId not in s.midiInst:
            s.midiInst[instId] = (s.pid, s.vid, chan, midi, s.midprg[2], s.midprg[3])
    addElemT(nt, 'voice', '1', lev + 1)
    if noMsrRest:
        addElemT(nt, 'type', xmltype, lev + 1)
    for i in range(ndot):
        dot = E.Element('dot')
        addElem(nt, dot, lev + 1)
    decos = s.getNoteDecos(n)
    if acc and (not tstop):
        e = E.Element('accidental')
        if 'courtesy' in decos:
            e.set('parentheses', 'yes')
            decos.remove('courtesy')
        e.text = s.accTab[acc]
        addElem(nt, e, lev + 1)
    tupnotation = ''
    if s.ntup >= 0:
        tmod = mkTmod(s.tmnum, s.tmden, lev + 1)
        addElem(nt, tmod, lev + 1)
        if s.ntup > 0 and (not s.tupnts):
            tupnotation = 'start'
        s.tupnts.append((rdvs, tmod))
        if s.ntup == 0:
            if rdvs:
                tupnotation = 'stop'
            s.cmpNormType(rdvs, lev + 1)
    hasStem = 1
    if not ischord:
        s.chordDecos = {}
    if 'stemless' in decos or (s.nostems and n.name != 'rest') or 'stemless' in s.chordDecos:
        hasStem = 0
        addElemT(nt, 'stem', 'none', lev + 1)
        if 'stemless' in decos:
            decos.remove('stemless')
        if hasattr(n, 'pitches'):
            s.chordDecos['stemless'] = 1
    if notehead:
        nh = addElemT(nt, 'notehead', re.sub('[+-]$', '', notehead), lev + 1)
        if notehead[-1] in '+-':
            nh.set('filled', 'yes' if notehead[-1] == '+' else 'no')
    gstaff = s.gStaffNums.get(s.vid, 0)
    if gstaff:
        addElemT(nt, 'staff', str(gstaff), lev + 1)
    if hasStem:
        s.doBeams(n, nt, den, lev + 1)
    s.doNotations(n, decos, ptup, alter, tupnotation, tstop, nt, lev + 1)
    if n.objs:
        s.doLyr(n, nt, lev + 1)
    else:
        s.prevLyric = {}
    return nt