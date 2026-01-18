from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def readPfmt(x, n):
    if not s.pageFmtAbc:
        s.pageFmtAbc = s.pageFmtDef
    ro = re.search('[^.\\d]*([\\d.]+)\\s*(cm|in|pt)?', x)
    if ro:
        x, unit = ro.groups()
        u = {'cm': 10.0, 'in': 25.4, 'pt': 25.4 / 72}[unit] if unit else 1.0
        s.pageFmtAbc[n] = float(x) * u
    else:
        info('error in page format: %s' % x)