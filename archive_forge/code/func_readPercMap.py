from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def readPercMap(x):

    def getMidNum(sndnm):
        pnms = sndnm.split('-')
        ps = s.percsnd[:]
        _f = lambda ip, xs, pnm: ip < len(xs) and xs[ip].find(pnm) > -1
        for ip, pnm in enumerate(pnms):
            ps = [(nm, mnum) for nm, mnum in ps if _f(ip, nm.split('-'), pnm)]
            if len(ps) <= 1:
                break
        if len(ps) == 0:
            info('drum sound: %s not found' % sndnm)
            return '38'
        return ps[0][1]

    def midiVal(acc, step, oct):
        oct = (4 if step.upper() == step else 5) + int(oct)
        return oct * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(step.upper())] + {'^': 1, '_': -1, '=': 0}.get(acc, 0) + 12
    p0, p1, p2, p3, p4 = abc_percmap.parseString(x).asList()
    acc, astep, aoct = p1
    nstep, noct = (astep, aoct) if p2 == '*' else p2
    if p3 == '*':
        midi = str(midiVal(acc, astep, aoct))
    elif isinstance(p3, list_type):
        midi = str(midiVal(p3[0], p3[1], p3[2]))
    elif isinstance(p3, int_type):
        midi = str(p3)
    else:
        midi = getMidNum(p3.lower())
    head = re.sub('(.)-([^x])', '\\1 \\2', p4)
    s.percMap[s.pid, acc + astep, aoct] = (nstep, noct, midi, head)