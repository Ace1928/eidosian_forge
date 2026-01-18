from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def convertBroken(t):
    prev = None
    brk = ''
    remove = []
    for i, x in enumerate(t):
        if x.name == 'note' or x.name == 'chord' or x.name == 'rest':
            if brk:
                doBroken(prev, brk, x)
                brk = ''
            else:
                prev = x
        elif x.name == 'broken':
            brk = x.t[0]
            remove.insert(0, i)
    for i in remove:
        del t[i]