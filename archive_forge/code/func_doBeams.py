from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doBeams(s, n, nt, den, lev):
    if hasattr(n, 'chord') or hasattr(n, 'grace'):
        s.grcbbrk = s.grcbbrk or n.bbrk.t[0]
        return
    bbrk = s.grcbbrk or n.bbrk.t[0] or den < 32
    s.grcbbrk = False
    if not s.prevNote:
        pbm = None
    else:
        pbm = s.prevNote.find('beam')
    bm = E.Element('beam', number='1')
    bm.text = 'begin'
    if pbm != None:
        if bbrk:
            if pbm.text == 'begin':
                s.prevNote.remove(pbm)
            elif pbm.text == 'continue':
                pbm.text = 'end'
            s.prevNote = None
        else:
            bm.text = 'continue'
    if den >= 32 and n.name != 'rest':
        addElem(nt, bm, lev)
        s.prevNote = nt