from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def setlines(s, stflines, stfnum):
    if stfnum != s.curstaff:
        s.curstaff = stfnum
        s.snaarVrij = []
        for i in range(stflines):
            s.snaarVrij.append([])
        s.beginZoek()