from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
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