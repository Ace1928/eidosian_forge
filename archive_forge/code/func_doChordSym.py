from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doChordSym(s, maat, sym, lev):
    alterMap = {'#': '1', '=': '0', 'b': '-1'}
    rnt = sym.root.t
    chord = E.Element('harmony')
    addElem(maat, chord, lev)
    root = E.Element('root')
    addElem(chord, root, lev + 1)
    addElemT(root, 'root-step', rnt[0], lev + 2)
    if len(rnt) == 2:
        addElemT(root, 'root-alter', alterMap[rnt[1]], lev + 2)
    kind = s.chordTab.get(sym.kind.t[0], 'major') if sym.kind.t else 'major'
    addElemT(chord, 'kind', kind, lev + 1)
    if hasattr(sym, 'bass'):
        bnt = sym.bass.t
        bass = E.Element('bass')
        addElem(chord, bass, lev + 1)
        addElemT(bass, 'bass-step', bnt[0], lev + 2)
        if len(bnt) == 2:
            addElemT(bass, 'bass-alter', alterMap[bnt[1]], lev + 2)
    degs = getattr(sym, 'degree', '')
    if degs:
        if type(degs) != list_type:
            degs = [degs]
        for deg in degs:
            deg = deg.t[0]
            if deg[0] == '#':
                alter = '1'
                deg = deg[1:]
            elif deg[0] == 'b':
                alter = '-1'
                deg = deg[1:]
            else:
                alter = '0'
                deg = deg
            degree = E.Element('degree')
            addElem(chord, degree, lev + 1)
            addElemT(degree, 'degree-value', deg, lev + 2)
            addElemT(degree, 'degree-alter', alter, lev + 2)
            addElemT(degree, 'degree-type', 'add', lev + 2)