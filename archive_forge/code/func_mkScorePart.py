from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkScorePart(s, id, vids_p, partAttr, lev):

    def mkInst(instId, vid, midchan, midprog, midnot, vol, pan, lev):
        si = E.Element('score-instrument', id=instId)
        pnm = partAttr.get(vid, [''])[0]
        addElemT(si, 'instrument-name', pnm or 'dummy', lev + 2)
        mi = E.Element('midi-instrument', id=instId)
        if midchan:
            addElemT(mi, 'midi-channel', midchan, lev + 2)
        if midprog:
            addElemT(mi, 'midi-program', str(int(midprog) + 1), lev + 2)
        if midnot:
            addElemT(mi, 'midi-unpitched', str(int(midnot) + 1), lev + 2)
        if vol:
            addElemT(mi, 'volume', '%.2f' % (int(vol) / 1.27), lev + 2)
        if pan:
            addElemT(mi, 'pan', '%.2f' % (int(pan) / 127.0 * 180 - 90), lev + 2)
        return (si, mi)
    naam, subnm, midprg = partAttr[id]
    sp = E.Element('score-part', id='P' + id)
    nm = E.Element('part-name')
    nm.text = naam
    addElem(sp, nm, lev + 1)
    snm = E.Element('part-abbreviation')
    snm.text = subnm
    if subnm:
        addElem(sp, snm, lev + 1)
    inst = []
    for instId, (pid, vid, chan, midprg, vol, pan) in sorted(s.midiInst.items()):
        midprg, midnot = ('0', midprg) if chan == '10' else (midprg, '')
        if pid == id:
            inst.append(mkInst(instId, vid, chan, midprg, midnot, vol, pan, lev))
    for si, mi in inst:
        addElem(sp, si, lev + 1)
    for si, mi in inst:
        addElem(sp, mi, lev + 1)
    return sp