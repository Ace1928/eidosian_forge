from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def instChange(midchan, midprog):
    if midchan and midchan != s.midprg[0]:
        instDir('midi-channel', midchan, 'chan: %s')
    if midprog and midprog != s.midprg[1]:
        instDir('midi-program', str(int(midprog) + 1), 'prog: %s')