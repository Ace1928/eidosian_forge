from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def bezet(s, snaar, t1, t2):
    xs = s.snaarVrij[snaar]
    for i, (tb, te) in enumerate(xs):
        if t1 >= te:
            continue
        xs.insert(i, (t1, t2))
        return
    xs.append((t1, t2))