from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def getNoteDecos(s, n):
    decos = s.nextdecos
    ndeco = getattr(n, 'deco', 0)
    if ndeco:
        decos += [s.usrSyms.get(d, d).strip('!+') for d in ndeco.t]
    s.nextdecos = []
    if s.tabStaff == s.pid and s.fOpt and (n.name != 'rest'):
        if [d for d in decos if d in '0123456789'] == []:
            decos.append('0')
    return decos