from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doFields(s, maat, fieldmap, lev):

    def instDir(midelm, midnum, dirtxt):
        instId = 'I%s-%s' % (s.pid, s.vid)
        words = E.Element('words')
        words.text = dirtxt % midnum
        snd = E.Element('sound')
        mi = E.Element('midi-instrument', id=instId)
        dir = addDirection(maat, words, lev, gstaff, placement='above')
        addElem(dir, snd, lev + 1)
        addElem(snd, mi, lev + 2)
        addElemT(mi, midelm, midnum, lev + 3)

    def addTrans(n):
        e = E.Element('transpose')
        addElemT(e, 'chromatic', n, lev + 2)
        atts.append((9, e))

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
    s.diafret = 0
    atts = []
    gstaff = s.gStaffNums.get(s.vid, 0)
    for ftype, field in fieldmap.items():
        if not field:
            continue
        if ftype == 'Div':
            d = E.Element('divisions')
            d.text = field
            atts.append((1, d))
        elif ftype == 'gstaff':
            e = E.Element('staves')
            e.text = str(field)
            atts.append((4, e))
        elif ftype == 'M':
            if field == 'none':
                continue
            if field == 'C':
                field = '4/4'
            elif field == 'C|':
                field = '2/2'
            t = E.Element('time')
            if '/' not in field:
                info('M:%s not recognized, 4/4 assumed' % field)
                field = '4/4'
            beats, btype = field.split('/')[:2]
            try:
                s.mdur = simplify(eval(beats), int(btype))
            except:
                info('error in M:%s, 4/4 assumed' % field)
                s.mdur = (4, 4)
                beats, btype = ('4', '4')
            addElemT(t, 'beats', beats, lev + 2)
            addElemT(t, 'beat-type', btype, lev + 2)
            atts.append((3, t))
        elif ftype == 'K':
            accs = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
            mode = ''
            key = re.match('\\s*([A-G][#b]?)\\s*([a-zA-Z]*)', field)
            alts = re.search('\\s((\\s?[=^_][A-Ga-g])+)', ' ' + field)
            if key:
                key, mode = key.groups()
                mode = mode.lower()[:3]
                if mode not in s.offTab:
                    mode = 'maj'
                fifths = s.sharpness.index(key) - s.offTab[mode]
                if fifths >= 0:
                    s.keyAlts = dict(zip(accs[:fifths], fifths * ['1']))
                else:
                    s.keyAlts = dict(zip(accs[fifths:], -fifths * ['-1']))
            elif field.startswith('none') or field == '':
                fifths = 0
                mode = 'maj'
            if alts:
                alts = re.findall('[=^_][A-Ga-g]', alts.group(1))
                alts = [(x[1], s.alterTab[x[0]]) for x in alts]
                for step, alter in alts:
                    s.keyAlts[step.upper()] = alter
                k = E.Element('key')
                koctave = []
                lowerCaseSteps = [step.upper() for step, alter in alts if step.islower()]
                for step, alter in sorted(list(s.keyAlts.items())):
                    if alter == '0':
                        del s.keyAlts[step.upper()]
                        continue
                    addElemT(k, 'key-step', step.upper(), lev + 2)
                    addElemT(k, 'key-alter', alter, lev + 2)
                    koctave.append('5' if step in lowerCaseSteps else '4')
                if koctave:
                    for oct in koctave:
                        e = E.Element('key-octave', number=oct)
                        addElem(k, e, lev + 2)
                    atts.append((2, k))
            elif mode:
                k = E.Element('key')
                addElemT(k, 'fifths', str(fifths), lev + 2)
                addElemT(k, 'mode', s.modTab[mode], lev + 2)
                atts.append((2, k))
            doClef(field)
        elif ftype == 'L':
            try:
                s.unitLcur = lmap(int, field.split('/'))
            except:
                s.unitLcur = (1, 8)
            if len(s.unitLcur) == 1 or s.unitLcur[1] not in s.typeMap:
                info('L:%s is not allowed, 1/8 assumed' % field)
                s.unitLcur = (1, 8)
        elif ftype == 'V':
            doClef(field)
        elif ftype == 'I':
            s.doField_I(ftype, field, instDir, addTrans)
        elif ftype == 'Q':
            s.doTempo(maat, field, lev)
        elif ftype == 'P':
            words = E.Element('rehearsal')
            words.set('font-weight', 'bold')
            words.text = field
            addDirection(maat, words, lev, gstaff, placement='above')
        elif ftype in 'TCOAZNGHRBDFSU':
            info('**illegal header field in body: %s, content: %s' % (ftype, field))
        else:
            info('unhandled field: %s, content: %s' % (ftype, field))
    if atts:
        att = E.Element('attributes')
        addElem(maat, att, lev)
        for _, att_elem in sorted(atts, key=lambda x: x[0]):
            addElem(att, att_elem, lev + 1)
    if s.diafret:
        other = E.Element('other-direction')
        other.text = 'diatonic fretting'
        addDirection(maat, other, lev, 0)