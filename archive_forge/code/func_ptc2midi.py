from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def ptc2midi(n):
    pt = getattr(n, 'pitch', '')
    if pt:
        p = pt.t
        if len(p) == 3:
            acc, step, oct = p
        else:
            acc = ''
            step, oct = p
        nUp = step.upper()
        oct = (4 if nUp == step else 5) + int(oct)
        midi = oct * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(nUp)] + {'^': 1, '_': -1}.get(acc, 0) + 12
    else:
        midi = 130
    return midi