from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def isVrij(s, snaar, t1, t2):
    xs = s.snaarVrij[snaar]
    for i in range(s.snaarIx[snaar], len(xs)):
        tb, te = xs[i]
        if t1 >= te:
            continue
        if t1 >= tb:
            s.snaarIx[snaar] = i
            return 0
        if t2 > tb:
            s.snaarIx[snaar] = i
            return 0
        s.snaarIx[snaar] = i
        xs.insert(i, (t1, t2))
        return 1
    xs.append((t1, t2))
    s.snaarIx[snaar] = len(xs) - 1
    return 1