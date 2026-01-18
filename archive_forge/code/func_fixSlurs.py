from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def fixSlurs(x):

    def f(mo):
        n = int(mo.group(2))
        return (n * (mo.group(1) + '|'))[:-1]

    def g(mo):
        return mo.group(1).replace(' ', '')
    x = mm_rest.sub(f, x)
    x = bar_space.sub(g, x)
    return slur_move.sub('\\2\\1', x)