from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def decodeInput(data_string):
    try:
        enc = 'utf-8'
        unicode_string = data_string.decode(enc)
    except:
        try:
            enc = 'latin-1'
            unicode_string = data_string.decode(enc)
        except:
            raise ValueError('data not encoded in utf-8 nor in latin-1')
    info('decoded from %s' % enc)
    return unicode_string