from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doBroken(prev, brk, x):
    if not prev:
        info('error in broken rhythm: %s' % x)
        return
    nom1, den1 = prev.dur.t
    nom2, den2 = x.dur.t
    if brk == '>':
        nom1, den1 = simplify(3 * nom1, 2 * den1)
        nom2, den2 = simplify(1 * nom2, 2 * den2)
    elif brk == '<':
        nom1, den1 = simplify(1 * nom1, 2 * den1)
        nom2, den2 = simplify(3 * nom2, 2 * den2)
    elif brk == '>>':
        nom1, den1 = simplify(7 * nom1, 4 * den1)
        nom2, den2 = simplify(1 * nom2, 4 * den2)
    elif brk == '<<':
        nom1, den1 = simplify(1 * nom1, 4 * den1)
        nom2, den2 = simplify(7 * nom2, 4 * den2)
    else:
        return
    prev.dur.t = (nom1, den1)
    x.dur.t = (nom2, den2)