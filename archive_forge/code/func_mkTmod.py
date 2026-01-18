from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkTmod(tmnum, tmden, lev):
    tmod = E.Element('time-modification')
    addElemT(tmod, 'actual-notes', str(tmnum), lev + 1)
    addElemT(tmod, 'normal-notes', str(tmden), lev + 1)
    return tmod