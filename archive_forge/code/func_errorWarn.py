from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def errorWarn(line, loc, t):
    if not t[0]:
        return []
    info('**misplaced symbol: %s' % t[0], warn=0)
    lineCopy = line[:]
    if loc > 40:
        lineCopy = line[loc - 40:loc + 40]
        loc = 40
    info(lineCopy.replace('\n', ' '), warn=0)
    info(loc * '-' + '^', warn=0)
    return []