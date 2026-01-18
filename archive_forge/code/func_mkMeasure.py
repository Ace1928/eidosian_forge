from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkMeasure(s, i, t, lev, fieldmap={}):
    s.msreAlts = {}
    s.ntup, s.trem, s.intrem = (-1, 0, 0)
    s.acciatura = 0
    overlay = 0
    maat = E.Element('measure', number=str(i))
    if fieldmap:
        s.doFields(maat, fieldmap, lev + 1)
    if s.linebrk:
        e = E.Element('print')
        e.set('new-system', 'yes')
        addElem(maat, e, lev + 1)
        s.linebrk = 0
    for it, x in enumerate(t):
        if x.name == 'note' or x.name == 'rest':
            if x.dur.t[0] == 0:
                x.dur.t = tuple([1, x.dur.t[1]])
            note = s.mkNote(x, lev + 1)
            addElem(maat, note, lev + 1)
        elif x.name == 'lbar':
            bar = x.t[0]
            if bar == '|' or bar == '[|':
                pass
            elif ':' in bar:
                volta = x.t[1] if len(x.t) == 2 else ''
                s.mkBarline(maat, 'left', lev + 1, style='heavy-light', dir='forward', ending=volta)
            else:
                s.mkBarline(maat, 'left', lev + 1, ending=bar)
        elif x.name == 'rbar':
            bar = x.t[0]
            if bar == '.|':
                s.mkBarline(maat, 'right', lev + 1, style='dotted')
            elif ':' in bar:
                s.mkBarline(maat, 'right', lev + 1, style='light-heavy', dir='backward')
            elif bar == '||':
                s.mkBarline(maat, 'right', lev + 1, style='light-light')
            elif bar == '[|]' or bar == '[]':
                s.mkBarline(maat, 'right', lev + 1, style='none')
            elif '[' in bar or ']' in bar:
                s.mkBarline(maat, 'right', lev + 1, style='light-heavy')
            elif bar[0] == '&':
                overlay = 1
        elif x.name == 'tup':
            if len(x.t) == 3:
                n, into, nts = x.t
            elif len(x.t) == 2:
                n, into, nts = x.t + [0]
            else:
                n, into, nts = (x.t[0], 0, 0)
            if into == 0:
                into = 3 if n in [2, 4, 8] else 2
            if nts == 0:
                nts = n
            s.tmnum, s.tmden, s.ntup = (n, into, nts)
        elif x.name == 'deco':
            s.staffDecos(x.t, maat, lev + 1)
        elif x.name == 'text':
            pos, text = x.t[:2]
            place = 'above' if pos == '^' else 'below'
            words = E.Element('words')
            words.text = text
            gstaff = s.gStaffNums.get(s.vid, 0)
            addDirection(maat, words, lev + 1, gstaff, placement=place)
        elif x.name == 'inline':
            fieldtype, fieldval = (x.t[0], ' '.join(x.t[1:]))
            s.doFields(maat, {fieldtype: fieldval}, lev + 1)
        elif x.name == 'accia':
            s.acciatura = 1
        elif x.name == 'linebrk':
            s.supports_tag = 1
            if it > 0 and t[it - 1].name == 'lbar':
                e = E.Element('print')
                e.set('new-system', 'yes')
                addElem(maat, e, lev + 1)
            else:
                s.linebrk = 1
        elif x.name == 'chordsym':
            s.doChordSym(maat, x, lev + 1)
    s.stopBeams()
    s.prevmsre = maat
    return (maat, overlay)