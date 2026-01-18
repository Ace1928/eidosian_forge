from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def alignLyr(vce, lyrs):
    empty_el = pObj('leeg', '*')
    for k, lyr in enumerate(lyrs):
        i = 0
        for elem in vce:
            if elem.name == 'note' and (not (hasattr(elem, 'chord') or hasattr(elem, 'grace'))):
                if i >= len(lyr):
                    lr = empty_el
                else:
                    lr = lyr[i]
                lr.t[0] = lr.t[0].replace('%5d', ']')
                elem.objs.append(lr)
                if lr.name != 'sbar':
                    i += 1
            if elem.name == 'rbar' and i < len(lyr) and (lyr[i].name == 'sbar'):
                i += 1
    return vce