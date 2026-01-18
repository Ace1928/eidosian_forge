from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doArticulations(s, nt, nots, arts, lev):
    decos = []
    for a in arts:
        if a in s.artMap:
            art = E.Element('articulations')
            addElem(nots, art, lev)
            addElem(art, E.Element(s.artMap[a]), lev + 1)
        elif a in s.ornMap:
            orn = E.Element('ornaments')
            addElem(nots, orn, lev)
            addElem(orn, E.Element(s.ornMap[a]), lev + 1)
        elif a in ['trill(', 'trill)']:
            orn = E.Element('ornaments')
            addElem(nots, orn, lev)
            type = 'start' if a.endswith('(') else 'stop'
            if type == 'start':
                addElem(orn, E.Element('trill-mark'), lev + 1)
            addElem(orn, E.Element('wavy-line', type=type), lev + 1)
        elif a in s.tecMap:
            tec = E.Element('technical')
            addElem(nots, tec, lev)
            addElem(tec, E.Element(s.tecMap[a]), lev + 1)
        elif a in '0123456':
            tec = E.Element('technical')
            addElem(nots, tec, lev)
            if s.tabStaff == s.pid:
                alt = int(nt.findtext('pitch/alter') or 0)
                step = nt.findtext('pitch/step')
                oct = int(nt.findtext('pitch/octave'))
                midi = oct * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(step)] + alt + 12
                if a == '0':
                    firstFit = ''
                    for smid, istr in s.tunTup:
                        if midi >= smid:
                            isvrij = s.strAlloc.isVrij(istr - 1, s.gTime[0], s.gTime[1])
                            a = str(istr)
                            if not firstFit:
                                firstFit = a
                            if isvrij:
                                break
                    if not isvrij:
                        a = firstFit
                        s.strAlloc.bezet(int(a) - 1, s.gTime[0], s.gTime[1])
                else:
                    s.strAlloc.bezet(int(a) - 1, s.gTime[0], s.gTime[1])
                bmidi = s.tunmid[int(a) - 1]
                fret = midi - bmidi
                if fret < 25 and fret >= 0:
                    addElemT(tec, 'fret', str(fret), lev + 1)
                else:
                    altp = 'b' if alt == -1 else '#' if alt == 1 else ''
                    info('fret %d out of range, note %s%d on string %s' % (fret, step + altp, oct, a))
                addElemT(tec, 'string', a, lev + 1)
            else:
                addElemT(tec, 'fingering', a, lev + 1)
        else:
            decos.append(a)
    return decos