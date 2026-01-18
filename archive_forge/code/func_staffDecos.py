from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def staffDecos(s, decos, maat, lev):
    gstaff = s.gStaffNums.get(s.vid, 0)
    for d in decos:
        d = s.usrSyms.get(d, d).strip('!+')
        if d in s.dynaMap:
            dynel = E.Element('dynamics')
            addDirection(maat, dynel, lev, gstaff, [E.Element(d)], 'below', s.gcue_on)
        elif d in s.wedgeMap:
            if ')' in d:
                type = 'stop'
            else:
                type = 'crescendo' if '<' in d or 'crescendo' in d else 'diminuendo'
            addDirection(maat, E.Element('wedge', type=type), lev, gstaff)
        elif d.startswith('8v'):
            if 'a' in d:
                type, plce = ('down', 'above')
            else:
                type, plce = ('up', 'below')
            if ')' in d:
                type = 'stop'
            addDirection(maat, E.Element('octave-shift', type=type, size='8'), lev, gstaff, placement=plce)
        elif d in ['ped', 'ped-up']:
            type = 'stop' if d.endswith('up') else 'start'
            addDirection(maat, E.Element('pedal', type=type), lev, gstaff)
        elif d in ['coda', 'segno']:
            text, attr, val = s.capoMap[d]
            dir = addDirection(maat, E.Element(text), lev, gstaff, placement='above')
            sound = E.Element('sound')
            sound.set(attr, val)
            addElem(dir, sound, lev + 1)
        elif d in s.capoMap:
            text, attr, val = s.capoMap[d]
            words = E.Element('words')
            words.text = text
            dir = addDirection(maat, words, lev, gstaff, placement='above')
            sound = E.Element('sound')
            sound.set(attr, val)
            addElem(dir, sound, lev + 1)
        elif d == '(' or d == '.(':
            s.slurbeg.append(d)
        elif d in ['/-', '//-', '///-', '////-']:
            s.tmnum, s.tmden, s.ntup, s.trem, s.intrem = (2, 1, 2, len(d) - 1, 1)
        elif d in ['/', '//', '///']:
            s.trem = -len(d)
        else:
            s.nextdecos.append(d)