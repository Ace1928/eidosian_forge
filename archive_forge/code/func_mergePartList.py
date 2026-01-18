from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mergePartList(parts, rOpt, is_grand=0):

    def delAttrs(part):
        xs = [(m, e) for m in part.findall('measure') for e in m.findall('attributes')]
        for m, e in xs:
            for c in list(e):
                if c.tag == 'clef':
                    continue
                if c.tag == 'staff-details':
                    continue
                e.remove(c)
            if len(list(e)) == 0:
                m.remove(e)
    p1 = parts[0]
    for p2 in parts[1:]:
        if is_grand:
            delAttrs(p2)
        for i in range(len(p1) + 1, len(p2) + 1):
            maat = E.Element('measure', number=str(i))
            addElem(p1, maat, 2)
        slurs = p1.findall('measure/note/notations/slur')
        slur_max = max([int(slr.get('number')) for slr in slurs] + [0])
        vs = p1.findall('measure/note/voice')
        vnum_max = max([int(v.text) for v in vs] + [0])
        for im, m2 in enumerate(p2.findall('measure')):
            mergeMeasure(p1[im], m2, slur_max, vnum_max, rOpt, is_grand)
    return p1