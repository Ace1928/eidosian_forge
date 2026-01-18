from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def convertChord(t):
    ins = []
    for i, x in enumerate(t):
        if x.name == 'chord':
            if hasattr(x, 'rest') and (not hasattr(x, 'note')):
                if type(x.rest) == list_type:
                    x.rest = x.rest[0]
                ins.insert(0, (i, [x.rest]))
                continue
            num1, den1 = x.dur.t
            tie = getattr(x, 'tie', None)
            slurs = getattr(x, 'slurs', [])
            if type(x.note) != list_type:
                x.note = [x.note]
            elms = []
            j = 0
            nss = sorted(x.objs, key=ptc2midi, reverse=1) if mxm.orderChords else x.objs
            for nt in nss:
                if nt.name == 'note':
                    num2, den2 = nt.dur.t
                    nt.dur.t = simplify(num1 * num2, den1 * den2)
                    if tie:
                        nt.tie = tie
                    if j == 0 and slurs:
                        nt.slurs = slurs
                    if j > 0:
                        nt.chord = pObj('chord', [1])
                    else:
                        pitches = [n.pitch for n in x.note]
                        nt.pitches = pObj('pitches', pitches)
                    j += 1
                if nt.name not in ['dur', 'tie', 'slurs', 'rest']:
                    elms.append(nt)
            ins.insert(0, (i, elms))
    for i, notes in ins:
        for nt in reversed(notes):
            t.insert(i + 1, nt)
        del t[i]