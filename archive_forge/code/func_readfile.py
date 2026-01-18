from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def readfile(fnmext, errmsg='read error: '):
    try:
        if fnmext == '-.abc':
            fobj = stdin
        else:
            fobj = open(fnmext, 'rb')
        encoded_data = fobj.read()
        fobj.close()
        return encoded_data if type(encoded_data) == uni_type else decodeInput(encoded_data)
    except Exception as e:
        info(errmsg + repr(e) + ' ' + fnmext)
        return None