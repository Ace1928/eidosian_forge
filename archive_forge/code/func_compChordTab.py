from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def compChordTab():
    maj, min, aug, dim, dom, ch7, ch6, ch9, ch11, ch13, hd = 'major minor augmented diminished dominant -seventh -sixth -ninth -11th -13th half-diminished'.split()
    triad = zip('ma Maj maj M mi min m aug dim o + -'.split(), [maj, maj, maj, maj, min, min, min, aug, dim, dim, aug, min])
    seventh = zip('7 ma7 Maj7 M7 maj7 mi7 min7 m7 dim7 o7 -7 aug7 +7 m7b5 mi7b5'.split(), [dom, maj + ch7, maj + ch7, maj + ch7, maj + ch7, min + ch7, min + ch7, min + ch7, dim + ch7, dim + ch7, min + ch7, aug + ch7, aug + ch7, hd, hd])
    sixth = zip('6 ma6 M6 mi6 min6 m6'.split(), [maj + ch6, maj + ch6, maj + ch6, min + ch6, min + ch6, min + ch6])
    ninth = zip('9 ma9 M9 maj9 Maj9 mi9 min9 m9'.split(), [dom + ch9, maj + ch9, maj + ch9, maj + ch9, maj + ch9, min + ch9, min + ch9, min + ch9])
    elevn = zip('11 ma11 M11 maj11 Maj11 mi11 min11 m11'.split(), [dom + ch11, maj + ch11, maj + ch11, maj + ch11, maj + ch11, min + ch11, min + ch11, min + ch11])
    thirt = zip('13 ma13 M13 maj13 Maj13 mi13 min13 m13'.split(), [dom + ch13, maj + ch13, maj + ch13, maj + ch13, maj + ch13, min + ch13, min + ch13, min + ch13])
    sus = zip('sus sus4 sus2'.split(), ['suspended-fourth', 'suspended-fourth', 'suspended-second'])
    return dict(list(triad) + list(seventh) + list(sixth) + list(ninth) + list(elevn) + list(thirt) + list(sus))