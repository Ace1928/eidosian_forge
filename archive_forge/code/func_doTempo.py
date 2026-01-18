from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doTempo(s, maat, field, lev):
    gstaff = s.gStaffNums.get(s.vid, 0)
    t = re.search('(\\d)/(\\d\\d?)\\s*=\\s*(\\d[.\\d]*)|(\\d[.\\d]*)', field)
    rtxt = re.search('"([^"]*)"', field)
    if not t and (not rtxt):
        return
    elems = []
    if rtxt:
        num, den, upm = (1, 4, s.tempoMap.get(rtxt.group(1).lower().strip(), 120))
        words = E.Element('words')
        words.text = rtxt.group(1)
        elems.append((words, []))
    if t:
        try:
            if t.group(4):
                num, den, upm = (1, s.unitLcur[1], float(t.group(4)))
            else:
                num, den, upm = (int(t.group(1)), int(t.group(2)), float(t.group(3)))
        except:
            info('conversion error: %s' % field)
            return
        num, den = simplify(num, den)
        dotted, den_not = (1, den // 2) if num == 3 else (0, den)
        metro = E.Element('metronome')
        u = E.Element('beat-unit')
        u.text = s.typeMap.get(4 * den_not, 'quarter')
        pm = E.Element('per-minute')
        pm.text = ('%.2f' % upm).rstrip('0').rstrip('.')
        subelms = [u, E.Element('beat-unit-dot'), pm] if dotted else [u, pm]
        elems.append((metro, subelms))
    dir = addDirection(maat, elems, lev, gstaff, [], placement='above')
    if num != 1 and num != 3:
        info('in Q: numerator in %d/%d not supported' % (num, den))
    qpm = 4.0 * num * upm / den
    sound = E.Element('sound')
    sound.set('tempo', '%.2f' % qpm)
    addElem(dir, sound, lev + 1)