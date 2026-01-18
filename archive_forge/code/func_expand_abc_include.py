from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def expand_abc_include(abctxt):
    ys = []
    for x in abctxt.splitlines():
        if x.startswith('%%abc-include') or x.startswith('I:abc-include'):
            x = readfile(x[13:].strip(), 'include error: ')
        if x != None:
            ys.append(x)
    return '\n'.join(ys)