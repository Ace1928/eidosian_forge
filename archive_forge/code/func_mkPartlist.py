from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkPartlist(s, vids, partAttr, lev):

    def addPartGroup(sym, num):
        pg = E.Element('part-group', number=str(num), type='start')
        addElem(partlist, pg, lev + 1)
        addElemT(pg, 'group-symbol', sym, lev + 2)
        addElemT(pg, 'group-barline', 'yes', lev + 2)
    partlist = E.Element('part-list')
    g_num = 0
    for g in s.groups or vids:
        if g == '[':
            g_num += 1
            addPartGroup('bracket', g_num)
        elif g == '{':
            g_num += 1
            addPartGroup('brace', g_num)
        elif g in '}]':
            pg = E.Element('part-group', number=str(g_num), type='stop')
            addElem(partlist, pg, lev + 1)
            g_num -= 1
        else:
            if g not in vids:
                continue
            sp = s.mkScorePart(g, vids, partAttr, lev + 1)
            addElem(partlist, sp, lev + 1)
    return partlist