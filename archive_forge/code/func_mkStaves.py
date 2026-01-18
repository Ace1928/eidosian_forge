from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkStaves(p, vdefs):
    xs = []
    for i, x in enumerate(p.objs):
        if type(x) == pObj:
            us = mkStaves(x, vdefs)
            if x.name == 'voicegr':
                xs.append(us)
                vids = [y.objs[0] for y in x.objs]
                vdefs = setFristVoiceNameFromGroup(vids, vdefs)
                p.objs[i] = x.objs[0]
            else:
                xs.extend(us)
        elif p.t[0] not in '{*':
            xs.append(p.t[0])
    return xs