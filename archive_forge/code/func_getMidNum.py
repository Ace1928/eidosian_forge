from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def getMidNum(sndnm):
    pnms = sndnm.split('-')
    ps = s.percsnd[:]
    _f = lambda ip, xs, pnm: ip < len(xs) and xs[ip].find(pnm) > -1
    for ip, pnm in enumerate(pnms):
        ps = [(nm, mnum) for nm, mnum in ps if _f(ip, nm.split('-'), pnm)]
        if len(ps) <= 1:
            break
    if len(ps) == 0:
        info('drum sound: %s not found' % sndnm)
        return '38'
    return ps[0][1]